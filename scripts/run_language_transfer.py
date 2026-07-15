"""Paper-2 RQ1 (EN axis) — language-transfer extraction.

Same Polish entities, ENGLISH question templates. The entity strings are
deliberately UNCHANGED (Polish names/places): the manipulation is the language
of the question stem only, mirroring the neutral-template control design
(results/<slug>/domains/cities_neutral). Templates (constants, mirroring the
Polish ones):

    PEOPLE (athletes/writers/musicians): "Who is {entity}? Answer in one sentence."
    PLACES (cities):                     "What is {entity}? Answer in one sentence."

For ONE model (env BIELIK_MODEL_ID) it extracts all four domains:
  - athletes  -> domains/athletes_en/   (entities from data/<slug>/labeled.parquet)
  - cities    -> domains/cities_en/     (entities from candidates_domains.DOMAINS)
  - writers   -> domains/writers_en/
  - musicians -> domains/musicians_en/

Outputs per job (condition-only labels, forward-only — extended metrics are
intentionally skipped, matching run_template_control.py):
  data/<slug>/domains/<domain>_en/labeled.parquet
  results/<slug>/domains/<domain>_en/{signals.parquet, hidden_states.npz}

Every job verifies 126 rows, npz loads with allow_pickle=False, no NaNs in
metric columns, and spot-checks that the entity-token index resolves on a
minimal-token entity under THIS tokenizer (logs the decoded token) — the
positions.py leading-space fix must handle e.g. "Warszawa" directly after the
"is" of the EN stem.

Usage (ONE model job at a time — MPS single GPU):
    set -a; source .env; set +a
    BIELIK_MODEL_ID=speakleash/Bielik-1.5B-v3.0-Instruct \
        uv run python scripts/run_language_transfer.py [domain ...]
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
from bielik_hallu.dataset.label import render_prompt
from bielik_hallu.dataset.tokenization import tokenization_metadata
from bielik_hallu.extract.positions import find_entity_last_token_in_offsets
from bielik_hallu.extract.run import extract_signals

# English mirrors of the Polish templates ("Kim jest ...?" / "Czym jest ...?").
EN_TEMPLATE_PEOPLE = "Who is {entity}? Answer in one sentence."
EN_TEMPLATE_PLACES = "What is {entity}? Answer in one sentence."

# base domain -> (output slug, EN template)
JOBS = {
    "athletes": ("athletes_en", EN_TEMPLATE_PEOPLE),
    "cities": ("cities_en", EN_TEMPLATE_PLACES),
    "writers": ("writers_en", EN_TEMPLATE_PEOPLE),
    "musicians": ("musicians_en", EN_TEMPLATE_PEOPLE),
}
SIGNAL_METRIC_COLUMNS = ("ipr", "entropy", "first_token_entropy")
EXPECTED_ROWS = 126  # 42 KNOWN + 42 UNKNOWN_REAL + 42 FABRICATED


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _domain_entities(base_domain: str) -> list[tuple[str, str]]:
    """(condition, entity) pairs for a base domain.

    Athletes has no candidates_domains entry; its canonical entity lists live
    in the model-scoped labeled parquet (identical across models — verified:
    the 126 entities match between Bielik and Gemma parquets).
    """
    if base_domain == "athletes":
        df = pd.read_parquet(config.DATA_DIR / "labeled.parquet")
        return [(r["condition"], r["entity"]) for _, r in df.iterrows()]
    spec = DOMAINS[base_domain]
    return [(c, e) for c in config.CONDITIONS for e in spec[c]]


def build_en_labeled(base_domain: str, template: str, tokenizer, out_path: Path):
    """Condition-only labeled parquet under the EN template.

    Identical schema/labels to run_domains.build_domain_labeled; only the
    prompt language changes. Entities stay Polish on purpose.
    """
    rows = []
    for condition, entity in _domain_entities(base_domain):
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
    return df, out_path


def spot_check_entity_index(df: pd.DataFrame, template: str, tokenizer) -> None:
    """Resolve the entity-token index for a minimal-token entity and log it.

    Uses the exact encoding fed to the model (chat template + offsets), same
    path as extract_signals. Raises if the decoded token does not end the
    entity string.
    """
    row = df.loc[df["n_tokens_entity"].idxmin()]
    entity = row["entity"]
    rendered = render_prompt(tokenizer, entity, template)
    enc = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    idx = find_entity_last_token_in_offsets(offsets, rendered, entity)
    tok = tokenizer.decode(enc["input_ids"][0][idx])
    _log(f"  spot-check: entity={entity!r} (n_tokens={int(row['n_tokens_entity'])}) "
         f"-> token index {idx}, decoded token {tok!r}")
    if not entity.endswith(tok.strip()):
        raise ValueError(
            f"entity-token spot check failed: decoded token {tok!r} does not "
            f"end entity {entity!r} (index {idx})")


def _verify_signals(results_dir: Path, expected_rows_hidden: int) -> None:
    """Verify signals.parquet + hidden_states.npz — same guarantees as run_domains."""
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


def run_job(base_domain: str, tokenizer) -> None:
    t0 = time.time()
    out_slug, template = JOBS[base_domain]
    labeled_path = config.DATA_DIR / "domains" / out_slug / "labeled.parquet"
    results_dir = config.RESULTS_DIR / "domains" / out_slug
    _log(f"JOB {out_slug} (EN template) START | model={config.MODEL_SLUG}")
    df, _ = build_en_labeled(base_domain, template, tokenizer, labeled_path)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{labeled_path}: {len(df)} rows, expected {EXPECTED_ROWS}")
    _log(f"  built labeled: {len(df)} rows | "
         f"conditions={df['condition'].value_counts().to_dict()} | "
         f"template={template!r} | {labeled_path}")
    spot_check_entity_index(df, template, tokenizer)

    _log(f"  extracting signals -> {results_dir}")
    extract_signals(labeled_path, results_dir=results_dir, template=template)
    _verify_signals(results_dir, expected_rows_hidden=len(df))

    dt = time.time() - t0
    _log(f"JOB {out_slug} DONE | {len(df)} entities | {dt:.1f}s "
         f"({dt / 60:.1f} min) | outputs in {results_dir}")


def main() -> None:
    requested = sys.argv[1:] or list(JOBS.keys())
    for base in requested:
        if base not in JOBS:
            raise SystemExit(f"unknown base domain {base!r}; choose from {tuple(JOBS)}")

    _log(f"=== run_language_transfer: model={config.MODEL_ID} | "
         f"jobs={[JOBS[b][0] for b in requested]} ===")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    t0 = time.time()
    for base in requested:
        run_job(base, tokenizer)
    dt = time.time() - t0
    _log(f"=== run_language_transfer DONE | model={config.MODEL_SLUG} | "
         f"total {dt:.1f}s ({dt / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
