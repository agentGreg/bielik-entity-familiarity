"""Tokenizer confound check.

The v2 fabricated anchors (data/v2/entities_<domain>.parquet, built by
build_dataset_v2.py) were token-count matched to the real sample under the
BIELIK tokenizers only (1.5B + 11B, see bielik_hallu.dataset.v2.fabricate).
We check whether token length leaks the real/fabricated label
under the OTHER family tokenizers used in paper 2 (Gemma-4, Qwen3, PLLuM).

For each of the four family tokenizers (largest variant per family, as used
in the paper-2 fleet) we tokenize:
  * the bare entity string, and
  * the full v1-identical question prompt ("Kim jest {entity}? Odpowiedz
    jednym zdaniem." for people domains; "Czym jest ..." for cities),
and compute the real-vs-fabricated AUROC of the token count (fabricated =
positive class), per domain and pooled. Tokenizers only — no model weights.

Outputs: results/paper2_robustness/tokenizer_check.json.

Usage:
    uv run python scripts/p2_tokenizer_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bielik_hallu.dataset.v2.templates import templates_for  # noqa: E402

OUT_PATH = ROOT / "results" / "paper2_robustness" / "tokenizer_check.json"

DOMAINS = ("athletes", "cities", "writers", "musicians")

# Largest variant per family, matching the paper-2 fleet scripts.
TOKENIZERS = {
    "bielik": "speakleash/Bielik-11B-v3.0-Instruct",
    "gemma": "google/gemma-4-12b-it",
    "qwen": "Qwen/Qwen3-14B",
    "pllum": "CYFRAGOVPL/PLLuM-12B-instruct-2512",
}

# Verdict band: token length carries no exploitable signal if AUROC is here.
PASS_BAND = (0.35, 0.65)


def load_tokenizer(model_id: str):
    """Load a tokenizer, preferring the local HF cache (offline first)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id)


def rank_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUROC of pos > neg with average ranks for ties."""
    scores = np.concatenate([neg, pos])
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos, n_neg = len(pos), len(neg)
    u = ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def token_counts(tok, texts: list[str]) -> np.ndarray:
    return np.array([
        len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts
    ])


def main() -> None:
    frames = []
    for domain in DOMAINS:
        df = pd.read_parquet(ROOT / "data" / "v2" / f"entities_{domain}.parquet")
        df = df[["entity", "kind"]].copy()
        df["domain"] = domain
        df["prompt"] = df["entity"].map(
            lambda e, t=templates_for(domain)["prompt_pl"]: t.format(entity=e))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    n_real = int((data["kind"] == "real").sum())
    n_fab = int((data["kind"] == "fabricated").sum())
    print(f"entities: {len(data)} ({n_real} real, {n_fab} fabricated)")

    results: dict = {
        "n_real": n_real, "n_fab": n_fab,
        "pass_band": list(PASS_BAND),
        "tokenizers": {},
    }
    for family, model_id in TOKENIZERS.items():
        tok = load_tokenizer(model_id)
        fam_res: dict = {"model_id": model_id, "bare_entity": {}, "full_prompt": {}}
        for variant, col in (("bare_entity", "entity"), ("full_prompt", "prompt")):
            counts = token_counts(tok, data[col].tolist())
            per_domain = {}
            for domain in DOMAINS:
                m = data["domain"] == domain
                per_domain[domain] = round(rank_auroc(
                    counts[m & (data["kind"] == "real")],
                    counts[m & (data["kind"] == "fabricated")]), 4)
            pooled = round(rank_auroc(
                counts[(data["kind"] == "real").to_numpy()],
                counts[(data["kind"] == "fabricated").to_numpy()]), 4)
            worst = max(list(per_domain.values()) + [pooled],
                        key=lambda a: abs(a - 0.5))
            fam_res[variant] = {
                "pooled_auroc": pooled,
                "per_domain_auroc": per_domain,
                "worst_case_auroc": worst,
                "verdict": ("PASS" if PASS_BAND[0] <= worst <= PASS_BAND[1]
                            else "FLAG"),
            }
        results["tokenizers"][family] = fam_res
        print(f"[{family}] {model_id}")
        for variant in ("bare_entity", "full_prompt"):
            r = fam_res[variant]
            print(f"  {variant:12s} pooled={r['pooled_auroc']:.3f} "
                  f"per-domain={r['per_domain_auroc']} -> {r['verdict']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
