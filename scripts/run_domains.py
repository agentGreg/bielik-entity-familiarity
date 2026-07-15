"""Run E6 (cross-domain) + E7 (per-head) extraction for one model.

Usage:
    BIELIK_MODEL_ID=<hf-model-id> uv run python scripts/run_domains.py [domain ...]

With no positional args it runs all four jobs for the model in order:
    athletes  cities  writers  musicians

- ``athletes`` is a REFRESH job: it re-extracts on the existing
  ``data/<slug>/labeled.parquet`` (126 rows, 42/42/42) and OVERWRITES the
  top-level ``results/<slug>/{signals.parquet,hidden_states.npz,
  extended_signals.parquet,extended_report.json}`` (the earlier 123-row
  versions are superseded) and ADDS ``results/<slug>/attn_per_head.parquet``.
- ``cities`` / ``writers`` / ``musicians`` are NEW-domain jobs. They build a
  condition-only labeled parquet (no answer sampling, no judging:
  ``label_hallucination = 0`` for KNOWN, ``1`` otherwise) at
  ``data/<slug>/domains/<domain>/labeled.parquet`` and write all outputs into
  ``results/<slug>/domains/<domain>/``.

Schema of the new-domain labeled parquet:
    [entity, condition, prompt, label_hallucination, n_tokens_entity]
where ``prompt`` is rendered from the domain's own ``prompt_template`` (cities
use "Czym jest {entity}?...") and ``n_tokens_entity`` is that entity's token
count under THIS model's tokenizer.

Every job logs start/end timestamps and row counts and verifies its outputs
(expected row counts, npz loads with allow_pickle=False, no NaNs in metric
columns).
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from bielik_hallu import config
from bielik_hallu.analysis.report import build_extended_report
from bielik_hallu.dataset.candidates_domains import DOMAINS
from bielik_hallu.dataset.label import render_prompt
from bielik_hallu.dataset.tokenization import tokenization_metadata
from bielik_hallu.extract.extended import EXTENDED_METRICS, extract_extended_signals
from bielik_hallu.extract.run import extract_signals

NEW_DOMAINS = ("cities", "writers", "musicians")
ALL_JOBS = ("athletes", *NEW_DOMAINS)
# Metric columns that must never be NaN after extraction, per output file.
SIGNAL_METRIC_COLUMNS = ("ipr", "entropy", "first_token_entropy")
PER_HEAD_METRIC_COLUMNS = ("attn_entropy", "attn_entropy_norm")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def build_domain_labeled(domain: str, tokenizer, out_path: Path) -> Path:
    """Build a condition-only labeled parquet for a new domain.

    No answer sampling and no judging: label_hallucination is 0 for KNOWN and
    1 for UNKNOWN_REAL / FABRICATED. The prompt is rendered from the domain's
    own template and n_tokens_entity is measured with THIS model's tokenizer.
    """
    spec = DOMAINS[domain]
    template = spec["prompt_template"]
    rows = []
    for condition in config.CONDITIONS:
        for entity in spec[condition]:
            md = tokenization_metadata(tokenizer, entity)
            rows.append({
                "entity": entity,
                "condition": condition,
                "prompt": template.format(entity=entity),
                "label_hallucination": 0 if condition == "KNOWN" else 1,
                **md,
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return out_path


def _verify_signals(results_dir: Path, expected_rows_hidden: int) -> None:
    """Verify signals.parquet, hidden_states.npz and no-NaN metric columns."""
    sig = pd.read_parquet(results_dir / "signals.parquet")
    for col in SIGNAL_METRIC_COLUMNS:
        n_nan = int(sig[col].isna().sum())
        if n_nan:
            raise ValueError(f"{results_dir}/signals.parquet: {n_nan} NaNs in '{col}'")
    # npz must load without pickle (conditions stored as fixed-width unicode).
    with np.load(results_dir / "hidden_states.npz", allow_pickle=False) as npz:
        n = len(npz["labels"])
        if n != expected_rows_hidden:
            raise ValueError(
                f"{results_dir}/hidden_states.npz: labels has {n} rows, "
                f"expected {expected_rows_hidden}")
        if len(npz["conditions"]) != expected_rows_hidden:
            raise ValueError(f"{results_dir}/hidden_states.npz: conditions row mismatch")
    _log(f"  verify signals: {len(sig)} rows, npz labels={expected_rows_hidden}, "
         f"no NaNs in {SIGNAL_METRIC_COLUMNS}")


def _verify_extended(results_dir: Path, n_entities: int) -> None:
    """Verify extended_signals + attn_per_head parquet contents."""
    ext = pd.read_parquet(results_dir / "extended_signals.parquet")
    # effective_rank and logitlens_entropy are on every (layer, point) row and
    # must be finite; attn_entropy_* are NaN on the layer-N residual rows by
    # design, so they are checked only in the per-head file below.
    for col in ("effective_rank", "logitlens_entropy"):
        n_nan = int(ext[col].isna().sum())
        if n_nan:
            raise ValueError(
                f"{results_dir}/extended_signals.parquet: {n_nan} NaNs in '{col}'")

    ph = pd.read_parquet(results_dir / "attn_per_head.parquet")
    if list(ph.columns) != ["entity", "condition", "layer", "head", "point",
                            "attn_entropy", "attn_entropy_norm"]:
        raise ValueError(f"{results_dir}/attn_per_head.parquet: unexpected columns "
                         f"{list(ph.columns)}")
    for col in PER_HEAD_METRIC_COLUMNS:
        n_nan = int(ph[col].isna().sum())
        if n_nan:
            raise ValueError(
                f"{results_dir}/attn_per_head.parquet: {n_nan} NaNs in '{col}'")
    # Per entity, per point: n_attn_layers * n_heads rows. Cross-check the counts
    # are internally consistent (same head count on every layer, both points).
    n_layers = ph["layer"].nunique()
    n_heads = ph["head"].nunique()
    expected = n_entities * n_layers * n_heads * 2  # 2 measurement points
    if len(ph) != expected:
        raise ValueError(
            f"{results_dir}/attn_per_head.parquet: {len(ph)} rows, expected "
            f"{expected} (= {n_entities} entities x {n_layers} attn-layers x "
            f"{n_heads} heads x 2 points)")
    _log(f"  verify extended: extended_signals {len(ext)} rows; attn_per_head "
         f"{len(ph)} rows ({n_layers} layers x {n_heads} heads x 2 points x "
         f"{n_entities} entities), no NaNs")


def _report_summary(results_dir: Path) -> None:
    """Print best K-vs-F AUROC at prompt point per extended metric."""
    report = json.loads((results_dir / "extended_report.json").read_text())
    for metric in EXTENDED_METRICS:
        r = report["metrics"][metric]["prompt"]
        if r["best_layer"] is None:
            _log(f"  AUROC[prompt] {metric:24s}: no data")
        else:
            _log(f"  AUROC[prompt] {metric:24s}: {r['best_auroc']:.3f} "
                 f"@ layer {r['best_layer']}")


def run_job(job: str, tokenizer) -> None:
    t0 = time.time()
    if job == "athletes":
        labeled_path = config.DATA_DIR / "labeled.parquet"
        results_dir = config.RESULTS_DIR
        template = None  # athletes/people template is config.PROMPT_TEMPLATE
        if not labeled_path.exists():
            raise FileNotFoundError(
                f"{labeled_path} not found — athletes refresh needs the "
                "existing 126-row labeled parquet.")
        df = pd.read_parquet(labeled_path)
        _log(f"JOB athletes (refresh) START | model={config.MODEL_SLUG} | "
             f"labeled rows={len(df)} | "
             f"conditions={df['condition'].value_counts().to_dict()}")
    else:
        labeled_path = config.DATA_DIR / "domains" / job / "labeled.parquet"
        results_dir = config.RESULTS_DIR / "domains" / job
        template = DOMAINS[job]["prompt_template"]
        _log(f"JOB {job} (new domain) START | model={config.MODEL_SLUG}")
        build_domain_labeled(job, tokenizer, labeled_path)
        df = pd.read_parquet(labeled_path)
        _log(f"  built labeled: {len(df)} rows | "
             f"conditions={df['condition'].value_counts().to_dict()} | "
             f"template={template!r} | {labeled_path}")

    n_entities = len(df)

    # --- signals (ipr / entropy / first_token_entropy) + hidden states ---
    _log(f"  extracting signals -> {results_dir}")
    extract_signals(labeled_path, results_dir=results_dir, template=template)
    _verify_signals(results_dir, expected_rows_hidden=n_entities)

    # --- extended metrics + per-head attention (E7) ---
    _log(f"  extracting extended + per-head -> {results_dir}")
    ext_path = extract_extended_signals(labeled_path, results_dir=results_dir,
                                        template=template)
    sig = pd.read_parquet(ext_path)
    report = build_extended_report(sig, config.MODEL_ID)
    (results_dir / "extended_report.json").write_text(json.dumps(report, indent=2))
    _verify_extended(results_dir, n_entities=n_entities)
    _report_summary(results_dir)

    dt = time.time() - t0
    _log(f"JOB {job} DONE | {n_entities} entities | {dt:.1f}s "
         f"({dt / 60:.1f} min) | outputs in {results_dir}")


def main() -> None:
    requested = sys.argv[1:] or list(ALL_JOBS)
    for job in requested:
        if job not in ALL_JOBS:
            raise SystemExit(f"unknown job {job!r}; choose from {ALL_JOBS}")

    _log(f"=== run_domains: model={config.MODEL_ID} | jobs={requested} ===")
    # One tokenizer load for building new-domain parquets (n_tokens_entity).
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    t0 = time.time()
    for job in requested:
        run_job(job, tokenizer)
    dt = time.time() - t0
    _log(f"=== run_domains DONE | model={config.MODEL_SLUG} | jobs={requested} | "
         f"total {dt:.1f}s ({dt / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
