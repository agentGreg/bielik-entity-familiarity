"""Paper-2 Phase-2 extraction campaign over dataset v2 (popularity-graded).

Runs forward-only signal extraction for every (model, domain) pair over the
popularity-graded v2 entity lists (``data/v2/entities_<domain>.parquet``,
4 domains x 360 rows: 300 real deciles 0-9 + 60 fabricated decile -1).

Design
------
* **12 models**, smallest-first (see ``MODELS``). PL prompts only (justified in
  ``results/language_transfer.md``); templates come from
  ``bielik_hallu.dataset.v2.templates`` (people "Kim jest {entity}?...",
  cities "Czym jest {entity}?...").
* **Per (model, domain):** build a condition-free labeled parquet that carries
  the v2 popularity metadata (decile / pageviews / sitelinks / qid / kind)
  alongside the extraction contract columns (``condition``,
  ``label_hallucination``, ``prompt``, ``n_tokens_entity``), then run
  ``extract.run.extract_signals`` (forward-only): writes ``signals.parquet``
  (ipr / entropy / first_token_entropy per layer at BOTH the entity and prompt
  points) and ``hidden_states.npz`` (residual states at both points).
* Real entities -> condition ``REAL`` (label 0); fabricated -> ``FABRICATED``
  (label 1). Deciles/pageviews live in the labeled parquet so ``analyze_v2``
  can join popularity metadata back onto the per-entity signals by ``entity``.

Isolation & memory
------------------
Each (model, domain) job runs in its **own subprocess** (this file re-invoked
with ``--job <model_id> <domain>``). ``bielik_hallu.config`` reads
``BIELIK_MODEL_ID`` at import time, so the child sets the env var before import;
and because the whole process exits after each job, MPS memory is released
fully with no in-process ``del`` / ``gc`` / ``empty_cache`` bookkeeping needed.

Checkpointing
-------------
A job is considered DONE when ``results/<slug>/v2/<domain>/`` holds
``signals.parquet``, ``hidden_states.npz`` and ``job.json``. Completed jobs are
skipped on restart. Per-job runtime is logged and recorded in ``job.json``.

Failure policy
--------------
If a single model's job fails (OOM / arch / load error), the parent logs it,
records it in ``results/v2_campaign.json`` under ``failed``, and moves on — the
campaign does not stall.

Usage
-----
    # orchestrate the full campaign (parent):
    .venv/bin/python scripts/run_v2_campaign.py
    # optionally restrict models / domains:
    .venv/bin/python scripts/run_v2_campaign.py --models Bielik-1.5B Qwen3-1.7B
    .venv/bin/python scripts/run_v2_campaign.py --domains cities writers
    # single job (used internally by the parent; also runnable by hand):
    .venv/bin/python scripts/run_v2_campaign.py --job <hf-model-id> <domain>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_V2 = ROOT / "data" / "v2"
CAMPAIGN_JSON = ROOT / "results" / "v2_campaign.json"

# Ordered smallest-first for early signal. (label -> HF id). All cached locally.
MODELS: list[tuple[str, str]] = [
    ("Bielik-1.5B", "speakleash/Bielik-1.5B-v3.0-Instruct"),
    ("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    ("PLLuM-4B", "CYFRAGOVPL/PLLuM-4B-instruct-2512"),
    ("Qwen3-4B", "Qwen/Qwen3-4B"),
    ("Bielik-4.5B", "speakleash/Bielik-4.5B-v3.0-Instruct"),
    ("Bielik-Minitron-7B", "speakleash/Bielik-Minitron-7B-v3.0-Instruct"),
    ("Llama-PLLuM-8B", "CYFRAGOVPL/Llama-PLLuM-8B-instruct-2512"),
    ("Bielik-11B", "speakleash/Bielik-11B-v3.0-Instruct"),
    ("gemma-4-E4B", "google/gemma-4-E4B-it"),
    ("gemma-4-12b", "google/gemma-4-12b-it"),
    ("PLLuM-12B", "CYFRAGOVPL/PLLuM-12B-instruct-2512"),
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    # Base checkpoints for the controlled base-vs-PLLuM exposure comparison
    # (base-model control for the exposure comparison). Llama-3.1-8B is PLLuM-8B's base;
    # Mistral-NeMo-12B is PLLuM-12B's base — same architecture/tokenizer, so the
    # only difference is Polish continual pretraining.
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Mistral-Nemo-12B", "mistralai/Mistral-Nemo-Instruct-2407"),
]

DOMAINS = ("athletes", "cities", "writers", "musicians")

# Extraction contract: signals.parquet metric columns must never be NaN.
SIGNAL_METRIC_COLUMNS = ("ipr", "entropy", "first_token_entropy")
EXPECTED_ROWS = 360  # 300 real + 60 fabricated per domain


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def slug_for(model_id: str) -> str:
    return model_id.split("/")[-1]


def results_dir_for(model_id: str, domain: str) -> Path:
    return ROOT / "results" / slug_for(model_id) / "v2" / domain


def data_dir_for(model_id: str, domain: str) -> Path:
    return ROOT / "data" / slug_for(model_id) / "v2" / domain


def job_done(model_id: str, domain: str) -> bool:
    rd = results_dir_for(model_id, domain)
    return (
        (rd / "signals.parquet").exists()
        and (rd / "hidden_states.npz").exists()
        and (rd / "job.json").exists()
    )


# Labeled parquet column order: extraction-contract columns first, then the v2
# popularity metadata carried through for the analysis join.
LABELED_COLUMNS = [
    "entity", "condition", "prompt", "label_hallucination", "n_tokens_entity",
    "kind", "decile", "pageviews_12m", "sitelinks", "qid",
]


def build_v2_labeled(domain: str, tokenizer, entities=None):
    """Build the condition-free labeled table for a (model, domain).

    Pure/CPU-only: needs only a tokenizer (real or a whitespace fake) and the
    v2 entity table. Real entities -> condition ``REAL`` (label 0); fabricated
    -> ``FABRICATED`` (label 1). ``n_tokens_entity`` is the entity string's
    token count under THIS tokenizer; decile / pageviews / sitelinks / qid /
    kind are carried through so the analysis can join popularity metadata by
    ``entity``.
    """
    import pandas as pd

    from bielik_hallu.dataset.v2.templates import templates_for

    if entities is None:
        entities = pd.read_parquet(DATA_V2 / f"entities_{domain}.parquet")
    template = templates_for(domain)["prompt_pl"]

    rows = []
    for _, r in entities.iterrows():
        is_real = r["kind"] == "real"
        entity = r["entity"]
        n_tok = len(tokenizer(entity, add_special_tokens=False)["input_ids"])
        rows.append({
            "entity": entity,
            "condition": "REAL" if is_real else "FABRICATED",
            "prompt": template.format(entity=entity),
            "label_hallucination": 0 if is_real else 1,
            "n_tokens_entity": n_tok,
            "kind": r["kind"],
            "decile": int(r["decile"]),
            "pageviews_12m": int(r["pageviews_12m"]),
            "sitelinks": int(r["sitelinks"]),
            "qid": r["qid"],
        })
    return pd.DataFrame(rows, columns=LABELED_COLUMNS)


# ---------------------------------------------------------------------------
# Child (single job): build labeled parquet + extract + verify.
# Everything below the import guard runs in a subprocess with BIELIK_MODEL_ID
# already set, so ``bielik_hallu.config`` binds to the right model.
# ---------------------------------------------------------------------------
def run_single_job(model_id: str, domain: str) -> dict:
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    from bielik_hallu.dataset.v2.templates import templates_for
    from bielik_hallu.dataset.label import render_prompt
    from bielik_hallu.extract.run import extract_signals

    t0 = time.time()
    src = DATA_V2 / f"entities_{domain}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"missing dataset: {src}")

    template = templates_for(domain)["prompt_pl"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    labeled = build_v2_labeled(domain, tokenizer)

    data_dir = data_dir_for(model_id, domain)
    data_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = data_dir / "labeled.parquet"
    labeled.to_parquet(labeled_path)

    def n_tok(entity: str) -> int:
        return len(tokenizer(entity, add_special_tokens=False)["input_ids"])

    # Spot-check the LONGEST entity name: resolve its tokenization and log the
    # decoded last token so a mangled offset surfaces immediately.
    longest = labeled.loc[labeled["entity"].str.len().idxmax(), "entity"]
    rendered = render_prompt(tokenizer, longest, template)
    enc = tokenizer(rendered, return_offsets_mapping=True)
    from bielik_hallu.extract.positions import find_entity_last_token_in_offsets
    ent_idx = find_entity_last_token_in_offsets(enc["offset_mapping"], rendered, longest)
    last_tok = tokenizer.decode([enc["input_ids"][ent_idx]])
    _log(f"  spot-check longest entity {longest!r} "
         f"(n_tokens_entity={n_tok(longest)}): ent_idx={ent_idx} "
         f"last-token decoded={last_tok!r}")

    # Forward-only extraction (both points, all layers).
    results_dir = results_dir_for(model_id, domain)
    _log(f"  extract_signals -> {results_dir}")
    extract_signals(labeled_path, results_dir=results_dir, template=template)

    # --- verify ---
    sig = pd.read_parquet(results_dir / "signals.parquet")
    n_entities = sig["entity"].nunique()
    if n_entities != EXPECTED_ROWS:
        raise ValueError(
            f"{results_dir}/signals.parquet: {n_entities} unique entities, "
            f"expected {EXPECTED_ROWS}")
    for col in SIGNAL_METRIC_COLUMNS:
        n_nan = int(sig[col].isna().sum())
        if n_nan:
            raise ValueError(
                f"{results_dir}/signals.parquet: {n_nan} NaNs in '{col}'")
    with np.load(results_dir / "hidden_states.npz", allow_pickle=False) as npz:
        n_labels = len(npz["labels"])
        if n_labels != EXPECTED_ROWS:
            raise ValueError(
                f"{results_dir}/hidden_states.npz: labels has {n_labels} rows, "
                f"expected {EXPECTED_ROWS}")
        n_layer_keys = len([k for k in npz.files if k.startswith("entity_layer_")])

    dt = time.time() - t0
    meta = {
        "model_id": model_id,
        "model_slug": slug_for(model_id),
        "domain": domain,
        "rows": int(n_entities),
        "rows_ok": bool(n_entities == EXPECTED_ROWS),
        "hidden_layers": int(n_layer_keys),
        "longest_entity": longest,
        "longest_last_token": last_tok,
        "runtime_s": round(dt, 1),
        "finished_at": _ts(),
    }
    (results_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    _log(f"  JOB OK {slug_for(model_id)}/{domain} | rows={n_entities} | "
         f"layers={n_layer_keys} | {dt:.1f}s ({dt/60:.1f} min)")
    return meta


# ---------------------------------------------------------------------------
# Parent orchestration: spawn one child subprocess per job, checkpoint, log.
# ---------------------------------------------------------------------------
def _spawn_job(model_id: str, domain: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["BIELIK_MODEL_ID"] = model_id
    # Keep HF offline-friendly; all weights are cached.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--job", model_id, domain]
    return subprocess.run(cmd, env=env)


def orchestrate(models: list[tuple[str, str]], domains: tuple[str, ...]) -> None:
    _log(f"=== v2 campaign START | {len(models)} models x {len(domains)} domains "
         f"= {len(models) * len(domains)} jobs ===")
    t_all = time.time()
    summary = {"started_at": _ts(), "jobs": [], "failed": [], "skipped": []}
    if CAMPAIGN_JSON.exists():
        try:
            prev = json.loads(CAMPAIGN_JSON.read_text())
            summary["jobs"] = prev.get("jobs", [])
            summary["failed"] = prev.get("failed", [])
        except Exception:
            pass

    for label, model_id in models:
        for domain in domains:
            tag = f"{label}/{domain}"
            if job_done(model_id, domain):
                _log(f"SKIP (done) {tag}")
                summary["skipped"].append(tag)
                # Refresh job record from job.json in case we resumed.
                jj = results_dir_for(model_id, domain) / "job.json"
                try:
                    rec = json.loads(jj.read_text())
                    if not any(j.get("model_slug") == rec.get("model_slug")
                               and j.get("domain") == rec.get("domain")
                               for j in summary["jobs"]):
                        summary["jobs"].append(rec)
                except Exception:
                    pass
                _write_summary(summary)
                continue

            _log(f"JOB START {tag} | model_id={model_id}")
            t0 = time.time()
            proc = _spawn_job(model_id, domain)
            dt = time.time() - t0
            if proc.returncode == 0 and job_done(model_id, domain):
                rec = json.loads((results_dir_for(model_id, domain) / "job.json").read_text())
                summary["jobs"].append(rec)
                _log(f"JOB DONE {tag} | {dt:.1f}s")
            else:
                _log(f"JOB FAILED {tag} | rc={proc.returncode} | {dt:.1f}s "
                     f"— logging and continuing")
                summary["failed"].append({
                    "model": label, "model_id": model_id, "domain": domain,
                    "returncode": proc.returncode, "runtime_s": round(dt, 1),
                    "when": _ts(),
                })
            _write_summary(summary)

    summary["finished_at"] = _ts()
    summary["total_runtime_s"] = round(time.time() - t_all, 1)
    _write_summary(summary)
    dt = time.time() - t_all
    _log(f"=== v2 campaign DONE | {len(summary['jobs'])} jobs OK | "
         f"{len(summary['failed'])} failed | {dt/60:.1f} min ===")


def _write_summary(summary: dict) -> None:
    CAMPAIGN_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", nargs=2, metavar=("MODEL_ID", "DOMAIN"),
                    help="run a single (model, domain) job in-process (child)")
    ap.add_argument("--models", nargs="+", help="restrict to these model labels")
    ap.add_argument("--domains", nargs="+", help="restrict to these domains")
    args = ap.parse_args()

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    if args.job:
        model_id, domain = args.job
        run_single_job(model_id, domain)
        return

    models = MODELS
    if args.models:
        want = set(args.models)
        models = [m for m in MODELS if m[0] in want]
        if not models:
            raise SystemExit(f"no models matched {args.models}; "
                             f"choose from {[m[0] for m in MODELS]}")
    domains = tuple(args.domains) if args.domains else DOMAINS
    for d in domains:
        if d not in DOMAINS:
            raise SystemExit(f"unknown domain {d!r}; choose from {DOMAINS}")

    orchestrate(models, domains)


if __name__ == "__main__":
    main()
