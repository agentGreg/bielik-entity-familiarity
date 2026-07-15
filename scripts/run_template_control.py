"""Template-causality control.

The cities cross-domain transfer drop was attributed *correlationally* to the
prompt-template shift ("Czym jest X?" for places vs. "Kim jest X?" for people).
This script makes the attribution *causal* by re-extracting under a NEUTRAL
template that is grammatical for both people and places:

    NEUTRAL: "Co wiesz o {entity}? Odpowiedz jednym zdaniem."

For each Bielik model it re-extracts:
  - cities  under the neutral template -> results/<slug>/domains/cities_neutral/
  - writers under the neutral template -> results/<slug>/domains/writers_neutral/

The condition-only labels (KNOWN=0, else 1) and the 42/42/42 entity lists are
reused verbatim from ``candidates_domains.DOMAINS``; only the prompt template
changes. Forward-only extraction of signals.parquet + hidden_states.npz is what
the probe-transfer recomputation needs (extended metrics are not required for
Task A and are skipped to save GPU time).

Usage (ONE model job at a time — MPS single GPU):
    set -a; source .env; set +a
    BIELIK_MODEL_ID=speakleash/Bielik-1.5B-v3.0-Instruct \
        uv run python scripts/run_template_control.py

With no positional args it runs both base domains (cities, writers) for the
model in config.MODEL_ID. Pass a subset positionally, e.g. ``cities``.

Every job logs start/end timestamps + row counts and verifies its outputs
(expected row counts, npz loads with allow_pickle=False, no NaNs in metric
columns) — same guarantees as scripts/run_domains.py.
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from bielik_hallu import config
from bielik_hallu.dataset.candidates_domains import DOMAINS
from bielik_hallu.dataset.tokenization import tokenization_metadata
from bielik_hallu.extract.run import extract_signals

# Neutral template: grammatical for both people ("Kim jest") and places
# ("Czym jest"). This is the counterfactual stem that holds entity type fixed
# while removing the who/what asymmetry.
NEUTRAL_TEMPLATE = "Co wiesz o {entity}? Odpowiedz jednym zdaniem."

# base domain -> neutral output slug
JOBS = {
    "cities": "cities_neutral",
    "writers": "writers_neutral",
}
SIGNAL_METRIC_COLUMNS = ("ipr", "entropy", "first_token_entropy")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def build_neutral_labeled(base_domain: str, tokenizer, out_path: Path) -> Path:
    """Condition-only labeled parquet for a base domain under the neutral stem.

    Identical schema/labels to run_domains.build_domain_labeled, but the prompt
    is rendered from NEUTRAL_TEMPLATE instead of the domain's own template.
    """
    spec = DOMAINS[base_domain]
    rows = []
    for condition in config.CONDITIONS:
        for entity in spec[condition]:
            md = tokenization_metadata(tokenizer, entity)
            rows.append({
                "entity": entity,
                "condition": condition,
                "prompt": NEUTRAL_TEMPLATE.format(entity=entity),
                "label_hallucination": 0 if condition == "KNOWN" else 1,
                **md,
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df, out_path


def _verify_signals(results_dir: Path, expected_rows_hidden: int) -> None:
    sig = pd.read_parquet(results_dir / "signals.parquet")
    for col in SIGNAL_METRIC_COLUMNS:
        n_nan = int(sig[col].isna().sum())
        if n_nan:
            raise ValueError(f"{results_dir}/signals.parquet: {n_nan} NaNs in '{col}'")
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


def run_job(base_domain: str, out_slug: str, tokenizer) -> None:
    t0 = time.time()
    labeled_path = config.DATA_DIR / "domains" / out_slug / "labeled.parquet"
    results_dir = config.RESULTS_DIR / "domains" / out_slug
    _log(f"JOB {out_slug} (neutral template) START | model={config.MODEL_SLUG}")
    df, _ = build_neutral_labeled(base_domain, tokenizer, labeled_path)
    _log(f"  built labeled: {len(df)} rows | "
         f"conditions={df['condition'].value_counts().to_dict()} | "
         f"template={NEUTRAL_TEMPLATE!r} | {labeled_path}")

    n_entities = len(df)
    _log(f"  extracting signals -> {results_dir}")
    extract_signals(labeled_path, results_dir=results_dir, template=NEUTRAL_TEMPLATE)
    _verify_signals(results_dir, expected_rows_hidden=n_entities)

    dt = time.time() - t0
    _log(f"JOB {out_slug} DONE | {n_entities} entities | {dt:.1f}s "
         f"({dt / 60:.1f} min) | outputs in {results_dir}")


def main() -> None:
    requested = sys.argv[1:] or list(JOBS.keys())
    for base in requested:
        if base not in JOBS:
            raise SystemExit(f"unknown base domain {base!r}; choose from {tuple(JOBS)}")

    _log(f"=== run_template_control: model={config.MODEL_ID} | "
         f"jobs={[JOBS[b] for b in requested]} | template={NEUTRAL_TEMPLATE!r} ===")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    t0 = time.time()
    for base in requested:
        run_job(base, JOBS[base], tokenizer)
    dt = time.time() - t0
    _log(f"=== run_template_control DONE | model={config.MODEL_SLUG} | "
         f"total {dt:.1f}s ({dt / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
