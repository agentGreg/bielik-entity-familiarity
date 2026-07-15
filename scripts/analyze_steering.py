"""Paper-2 Phase 3 — analysis of the Gemma-4-12B causal steering sweep.

Reads results/paper2_steering/generations/*.jsonl (+ confirm/*.jsonl when
present) produced by scripts/steering_gemma.py and writes
results/paper2_steering/steering_report.{json,md} with:

  - baseline reproduction check (greedy alpha=0 vs stored sampled labels)
  - refusal-rate curves vs alpha per condition per direction, Wilson 95% CIs
  - H1 / H2 / H3 verdicts with deltas vs baseline and vs random-direction
    controls; dose-response monotonicity (Spearman over alpha levels)
  - degeneration guard: per-config rates of empty / repetitive / non-Polish
    generations (a direction that merely breaks the model is not "causal
    abstention")
  - claude-haiku-4-5 audit of 60 random steered generations
    (answer / refusal / hedged / gibberish) vs the marker-based metric
  - example generations per key config
  - direction metadata (cosines, norms) from directions_meta.json
  - honest limitations

Refusal metric: config.REFUSAL_MARKERS substring match (identical to
gemma_control.md / P0.1; no Gemma-specific additions exist in the repo — the
marker list reproduced the stored all_refusal labels exactly). The haiku audit
validates this metric under the distribution shift induced by steering.

Usage:
  set -a; source .env; set +a   # ANTHROPIC_API_KEY for the audit
  uv run python scripts/analyze_steering.py [--no-audit] [--n-audit 60]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BIELIK_MODEL_ID", "google/gemma-4-12b-it")
from bielik_hallu import config  # noqa: E402

OUT_DIR = ROOT / "results" / "paper2_steering"
GEN_DIR = OUT_DIR / "generations"
CONFIRM_DIR = OUT_DIR / "confirm"
AUDIT_PATH = OUT_DIR / "haiku_audit.jsonl"
REPORT_JSON = OUT_DIR / "steering_report.json"
REPORT_MD = OUT_DIR / "steering_report.md"
DIR_META = OUT_DIR / "directions_meta.json"

SEED = 0
AUDIT_MODEL = "claude-haiku-4-5"

POLISH_HINTS = re.compile(r"[ąćęłńóśźż]|(?:\b(?:jest|nie|to|się|który|która|polski|polska)\b)",
                          re.IGNORECASE)


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in config.REFUSAL_MARKERS)


# --------------------------------------------------------------------------
# degeneration heuristics
# --------------------------------------------------------------------------

def repetition_stats(text: str) -> tuple[float, float]:
    """(max_share, dup_share) over word trigrams.

    max_share = frequency share of the most common trigram (tight loops);
    dup_share = fraction of trigram occurrences that are repeats of an
    earlier trigram (catches long repeated phrases). Normal one-sentence
    answers sit near 0 on both.
    """
    words = text.split()
    if len(words) < 6:
        return 0.0, 0.0
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counts = Counter(tris)
    max_share = counts.most_common(1)[0][1] / len(tris)
    dup_share = sum(c - 1 for c in counts.values()) / len(tris)
    return max_share, dup_share


LATIN_EXT = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻàâäéèêëïîôöùûüçíáúýěščřžů]")


def degeneration_flags(text: str) -> dict:
    t = text.strip()
    max_share, dup_share = repetition_stats(t)
    words = t.split()
    # space-free token salad ("ymymym...", "НЕСНЕСНЕ") evades word-trigram
    # checks: catch it via extreme word lengths / non-Latin script share
    max_word = max((len(w) for w in words), default=0)
    letters = [c for c in t if c.isalpha()]
    nonlatin_share = (sum(1 for c in letters if not LATIN_EXT.match(c)) / len(letters)
                      if letters else 0.0)
    return {
        "empty": len(t) == 0,
        "repetitive": max_share > 0.3 or dup_share > 0.2,
        "repetition_ratio": round(max(max_share, dup_share), 3),
        "non_polish": len(t) > 20 and not POLISH_HINTS.search(t),
        "token_salad": max_word > 30 or nonlatin_share > 0.15,
        "n_words": len(words),
    }


def is_degenerate(flags: dict) -> bool:
    return (flags["empty"] or flags["repetitive"] or flags["non_polish"]
            or flags["token_salad"])


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_cell(sub: pd.DataFrame) -> dict:
    k, n = int(sub["is_refusal"].sum()), len(sub)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "refused": k, "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "degeneration_rate": round(float(sub["degenerate"].mean()), 4),
            "mean_words": round(float(sub["n_words"].mean()), 1)}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_generations() -> pd.DataFrame:
    rows = []
    for p in sorted(GEN_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["is_refusal"] = df["generated_text"].map(is_refusal)
    flags = df["generated_text"].map(degeneration_flags)
    for key in ("empty", "repetitive", "repetition_ratio", "non_polish",
                "token_salad", "n_words"):
        df[key] = flags.map(lambda f, k=key: f[k])
    df["degenerate"] = flags.map(is_degenerate)
    return df


def load_confirm() -> pd.DataFrame | None:
    if not CONFIRM_DIR.exists():
        return None
    rows = []
    for p in sorted(CONFIRM_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_stored_labels() -> pd.DataFrame:
    lab = pd.read_parquet(ROOT / "data" / "gemma-4-12b-it" / "labeled.parquet")
    per_ans = lab["answers"].apply(lambda arr: [is_refusal(a) for a in arr])
    return lab.assign(refusal_rate=per_ans.apply(np.mean),
                      refused=per_ans.apply(all))


# --------------------------------------------------------------------------
# haiku audit
# --------------------------------------------------------------------------

AUDIT_PROMPT = """You will see a Polish question about an entity and a model's answer. \
Classify the ANSWER into exactly one category:
- "answer": the model attempts a substantive answer about the entity
- "refusal": the model declines / says it does not know or has no information
- "hedged": the model gives partial information while flagging uncertainty or \
lack of knowledge
- "gibberish": the answer is degenerate (empty, repetitive loops, wrong \
language for no reason, or incoherent)

Question: {question}
Answer: {answer}

Reply with JSON only: {{"category": "<answer|refusal|hedged|gibberish>"}}"""


def run_audit(df: pd.DataFrame, n_audit: int) -> list[dict]:
    done = {}
    if AUDIT_PATH.exists():
        for line in AUDIT_PATH.read_text().splitlines():
            r = json.loads(line)
            done[(r["config_id"], r["entity"])] = r
    steered = df[df["kind"] != "baseline"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(steered), size=min(n_audit, len(steered)), replace=False)
    sample = steered.iloc[sorted(idx)]
    todo = [r for _, r in sample.iterrows()
            if (r["config_id"], r["entity"]) not in done]
    results = [done[(r["config_id"], r["entity"])] for _, r in sample.iterrows()
               if (r["config_id"], r["entity"]) in done]
    if todo:
        import anthropic
        client = anthropic.Anthropic()
        print(f"auditing {len(todo)} generations with {AUDIT_MODEL} "
              f"({len(results)} cached)")
        with open(AUDIT_PATH, "a") as fh:
            for r in todo:
                q = f"Kim jest {r['entity']}? Odpowiedz jednym zdaniem."
                msg = client.messages.create(
                    model=AUDIT_MODEL, max_tokens=64,
                    messages=[{"role": "user", "content": AUDIT_PROMPT.format(
                        question=q, answer=r["generated_text"] or "(empty)")}])
                raw = msg.content[0].text.strip()
                m = re.search(r'"category"\s*:\s*"(answer|refusal|hedged|gibberish)"', raw)
                cat = m.group(1) if m else "unparseable"
                rec = {"config_id": r["config_id"], "entity": r["entity"],
                       "condition": r["condition"], "alpha": r["alpha"],
                       "marker_refusal": bool(r["is_refusal"]),
                       "haiku_category": cat,
                       "generated_text": r["generated_text"]}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append(rec)
                time.sleep(0.2)
    return results


def audit_agreement(audit: list[dict]) -> dict:
    if not audit:
        return {"n": 0}
    cats = Counter(a["haiku_category"] for a in audit)
    # binary comparison: haiku refusal vs marker refusal (hedged counted as
    # non-refusal on both sides; reported separately)
    comparable = [a for a in audit if a["haiku_category"] in
                  ("answer", "refusal", "hedged", "gibberish")]
    agree = [a for a in comparable
             if (a["haiku_category"] == "refusal") == a["marker_refusal"]]
    disagreements = [
        {"config_id": a["config_id"], "entity": a["entity"],
         "marker_refusal": a["marker_refusal"], "haiku": a["haiku_category"],
         "text": a["generated_text"][:160]}
        for a in comparable
        if (a["haiku_category"] == "refusal") != a["marker_refusal"]]
    return {"n": len(audit), "haiku_categories": dict(cats),
            "binary_agreement": round(len(agree) / len(comparable), 4),
            "n_disagreements": len(disagreements),
            "disagreements": disagreements}


# --------------------------------------------------------------------------
# hypothesis synthesis
# --------------------------------------------------------------------------

def curves(df: pd.DataFrame) -> dict:
    """refusal-rate cells per direction-kind x layer x alpha x condition;
    baseline (alpha=0) is included in every direction family's curve."""
    out = {}
    base = df[df["kind"] == "baseline"]
    families = [("familiarity", 30), ("refusal", 44), ("random", 30), ("random", 44)]
    for kind, layer in families:
        fam = df[(df["kind"] == kind) & (df["layer"] == layer)]
        if fam.empty:
            continue
        key = f"{kind}_L{layer}"
        out[key] = {}
        for cond in ("KNOWN", "UNKNOWN_REAL", "FABRICATED"):
            pts = {}
            b = base[base["condition"] == cond]
            if len(b):
                pts["0"] = rate_cell(b)
            for alpha, sub in fam[fam["condition"] == cond].groupby("alpha"):
                if kind == "random":
                    for dk, s2 in sub.groupby("direction_kind"):
                        pts[f"{alpha:+g}/{dk}"] = rate_cell(s2)
                    pts[f"{alpha:+g}/pooled"] = rate_cell(sub)
                else:
                    pts[f"{alpha:+g}"] = rate_cell(sub)
            out[key][cond] = pts
    return out


def monotonicity(df: pd.DataFrame, kind: str, layer: int, cond: str) -> dict | None:
    from scipy.stats import spearmanr
    fam = df[(df["kind"] == kind) & (df["layer"] == layer) & (df["condition"] == cond)]
    base = df[(df["kind"] == "baseline") & (df["condition"] == cond)]
    if fam.empty:
        return None
    alphas, rates = [0.0], [base["is_refusal"].mean()]
    for a, sub in fam.groupby("alpha"):
        alphas.append(a)
        rates.append(sub["is_refusal"].mean())
    rho, p = spearmanr(alphas, rates)
    return {"alphas": alphas, "rates": [round(float(r), 4) for r in rates],
            "spearman_rho": round(float(rho), 4), "p": round(float(p), 4)}


def delta_vs_control(df: pd.DataFrame, kind: str, layer: int, alpha: float,
                     conds: list[str]) -> dict:
    """Refusal-rate delta of a steered config vs baseline and vs the random
    control at the same layer — amplitude-matched (same |alpha|) when such a
    control exists, else pooled over the layer's control alphas."""
    sub = df[(df["kind"] == kind) & (df["layer"] == layer)
             & (df["alpha"] == alpha) & (df["condition"].isin(conds))]
    if sub.empty:
        return {"missing": True}
    base = df[(df["kind"] == "baseline") & (df["condition"].isin(conds))]
    ctrl_all = df[(df["kind"] == "random") & (df["layer"] == layer)
                  & (df["condition"].isin(conds))]
    ctrl = ctrl_all[ctrl_all["alpha"].abs() == abs(alpha)]
    matched = len(ctrl) > 0
    if not matched:
        ctrl = ctrl_all
    cell, bcell = rate_cell(sub), rate_cell(base)
    out = {"config": cell, "baseline": bcell,
           "delta_vs_baseline": round(cell["rate"] - bcell["rate"], 4)}
    if len(ctrl):
        ccell = rate_cell(ctrl)
        out["random_control" + ("_amplitude_matched" if matched else "_pooled")] = ccell
        out["control_delta_vs_baseline"] = round(ccell["rate"] - bcell["rate"], 4)
    return out


def confabulation_curve(df: pd.DataFrame) -> dict:
    """Coherent non-refusal answers about UR/FAB entities per familiarity@L30
    alpha (0 = baseline config). A confabulation proxy for H2."""
    urfab = ["UNKNOWN_REAL", "FABRICATED"]
    out = {}
    b = df[(df["kind"] == "baseline") & (df["condition"].isin(urfab))]
    out["0"] = round(float((~b["is_refusal"] & ~b["degenerate"]).mean()), 4)
    fam = df[(df["kind"] == "familiarity") & (df["layer"] == 30)
             & (df["condition"].isin(urfab))]
    for a, s in fam.groupby("alpha"):
        out[f"{a:+g}"] = round(float((~s["is_refusal"] & ~s["degenerate"]).mean()), 4)
    return out


def random_coherent_subset(df: pd.DataFrame) -> dict:
    """Refusal rate among NON-degenerate random-control generations, with a
    condition-mix-adjusted baseline expectation.

    Interpretation guard: random perturbations suppress Polish refusal
    markers by corrupting the output (even the heuristically 'coherent'
    subset is visibly damaged), so a refusal-rate DROP under a trained
    direction is only evidence of steering when the outputs stay coherent —
    which the degeneration table establishes for the trained directions.
    """
    base = df[df["kind"] == "baseline"]
    br = base.groupby("condition")["is_refusal"].mean()
    out = {"baseline_refusal_rate": round(float(base["is_refusal"].mean()), 4)}
    for cid, s_all in df[df["kind"] == "random"].groupby("config_id"):
        s = s_all[~s_all["degenerate"]]
        if len(s) < 10:
            out[cid] = {"coherent_n": int(len(s)),
                        "note": "too few coherent generations"}
            continue
        k, n = int(s["is_refusal"].sum()), len(s)
        lo, hi = wilson_ci(k, n)
        exp = float(sum(br[c] * v for c, v in
                        s["condition"].value_counts().items()) / n)
        out[cid] = {"coherent_n": n, "refusal_rate": round(k / n, 4),
                    "wilson95": [round(lo, 4), round(hi, 4)],
                    "baseline_expected_for_mix": round(exp, 4)}
    return out


def verdicts() -> dict:
    """Plain-language hypothesis verdicts; numbers quoted from the computed
    report cells (curves / hypotheses / degeneration / confirmation)."""
    return {
        "H1": (
            "SUPPORTED. Pushing KNOWN toward 'unfamiliar' (familiarity@L30, "
            "alpha<0) increases refusals monotonically (Spearman rho ~ -0.99 "
            "across the 13-point dose curve): 0.238 baseline -> 0.357 (a=-8) "
            "-> 0.667 (a=-16) -> 1.000 (a=-24), all with 0% degeneration. "
            "Random directions never produce a single Polish refusal at any "
            "tested amplitude (2..24), so the refusal INCREASE is direction-"
            "specific. Sampled confirmation (5x, T=0.7, a=-24): mean "
            "refusal_rate 0.976, any-refusal 42/42 - not a greedy artifact."),
        "H2": (
            "SUPPORTED, with one interpretive guard. Pushing UR/FAB toward "
            "'familiar' (alpha>0) suppresses refusals monotonically: 0.726 "
            "baseline -> 0.393 (a=+4) -> 0.000 (a>=+8), 0% degeneration, and "
            "the confabulation proxy (coherent non-refusal answers about "
            "unknown/fabricated entities) rises 0.27 -> 1.00 - the model "
            "fluently invents biographies. Guard: random corruption ALSO "
            "zeroes refusal markers (coherent-subset refusal ~0 vs "
            "mix-expected ~0.5), so the marker drop alone is not specific; "
            "specificity rests on the steered outputs being 100% coherent "
            "Polish answers (degeneration table + haiku audit), which no "
            "random control achieves at any dose. Confirmation (a=+8): mean "
            "refusal_rate 0.019-0.024 vs stored 0.633-0.790."),
        "H3": (
            "PARTIALLY SUPPORTED / REFRAMED. The late-layer refusal "
            "direction (L44) is causally potent within UR/FAB, but "
            "asymmetrically: pushing AWAY from refusal is strong (0.726 -> "
            "0.357 at a=-12, 0% degeneration; confirmed at 5x T=0.7: mean "
            "refusal_rate ~0.36 vs stored ~0.71), while pushing TOWARD "
            "refusal adds only +0.05..+0.07 (baseline already 0.73-0.81 - "
            "ceiling), and on KNOWN it does nothing (0.238 -> 0.190..0.238). "
            "The strict premise - that the mid-band familiarity direction is "
            "NOT causally potent in UR/FAB - is falsified: familiarity@L30 "
            "saturates refusal in BOTH directions there (1.00 at a=-24, 0.00 "
            "at a>=+8). P0.1's correlational dissociation (familiarity does "
            "not RANK refusals within UR/FAB) therefore does not imply "
            "causal inertness; the familiarity axis remains causally "
            "upstream of refusal in all three conditions, and the late "
            "refusal representation adds an independently steerable, "
            "mostly-suppressive handle."),
    }


def pick_examples(df: pd.DataFrame, config_id: str, k: int = 3) -> list[dict]:
    sub = df[df["config_id"] == config_id]
    rng = np.random.default_rng(SEED)
    # prefer showing both refusals and answers when present
    ex = []
    for want_refusal in (True, False):
        s = sub[sub["is_refusal"] == want_refusal]
        if len(s):
            take = s.sample(n=min(2, len(s)), random_state=SEED)
            ex.extend(take.to_dict("records"))
    return [{"entity": e["entity"], "condition": e["condition"],
             "is_refusal": bool(e["is_refusal"]),
             "text": e["generated_text"][:280]} for e in ex[:k + 1]]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def build_report(df: pd.DataFrame, audit: list[dict], confirm: pd.DataFrame | None) -> dict:
    stored = load_stored_labels()
    base = df[df["kind"] == "baseline"].copy()
    base["stored_refused"] = base["entity"].map(stored.set_index("entity")["refused"])
    baseline_block = {
        "per_condition": {
            cond: {"greedy_refusal": rate_cell(sub),
                   "stored_all_refusal": int(sub["stored_refused"].sum()),
                   "stored_all_refusal_rate": round(float(sub["stored_refused"].mean()), 4)}
            for cond, sub in base.groupby("condition")},
        "per_entity_agreement_greedy_vs_stored": round(
            float((base["is_refusal"] == base["stored_refused"]).mean()), 4),
    }

    per_config = {}
    for cid, sub in df.groupby("config_id"):
        per_config[cid] = {
            "overall": rate_cell(sub),
            "per_condition": {c: rate_cell(s) for c, s in sub.groupby("condition")}}

    hyp = {
        "H1_known_toward_unfamiliar_increases_refusal": {
            "test": "familiarity@L30, alpha<0, KNOWN",
            "cells": {f"{a:+g}": delta_vs_control(df, "familiarity", 30, a, ["KNOWN"])
                      for a in (-2.0, -4.0, -8.0, -12.0, -16.0, -24.0)},
            "monotonicity_KNOWN": monotonicity(df, "familiarity", 30, "KNOWN"),
        },
        "H2_urfab_toward_familiar_decreases_refusal": {
            "test": "familiarity@L30, alpha>0, UR u FAB",
            "cells": {f"{a:+g}": delta_vs_control(df, "familiarity", 30, a,
                                                  ["UNKNOWN_REAL", "FABRICATED"])
                      for a in (2.0, 4.0, 8.0, 12.0, 16.0, 24.0)},
            "monotonicity_UR": monotonicity(df, "familiarity", 30, "UNKNOWN_REAL"),
            "monotonicity_FAB": monotonicity(df, "familiarity", 30, "FABRICATED"),
            # confabulation proxy: coherent (non-degenerate) non-refusal answer
            # about an UNKNOWN_REAL/FABRICATED entity
            "confabulation_rate_urfab": confabulation_curve(df),
        },
        "H3_late_refusal_direction_potent_in_urfab": {
            "test": "refusal@L44 vs familiarity@L30 within UR u FAB",
            "refusal_dir_cells": {f"{a:+g}": delta_vs_control(
                df, "refusal", 44, a, ["UNKNOWN_REAL", "FABRICATED"])
                for a in (-12.0, -6.0, -3.0, 3.0, 6.0, 12.0)},
            "familiarity_dir_cells_urfab": {f"{a:+g}": delta_vs_control(
                df, "familiarity", 30, a, ["UNKNOWN_REAL", "FABRICATED"])
                for a in (-24.0, -8.0, 8.0, 24.0)},
            "monotonicity_UR": monotonicity(df, "refusal", 44, "UNKNOWN_REAL"),
            "monotonicity_FAB": monotonicity(df, "refusal", 44, "FABRICATED"),
            "refusal_dir_on_KNOWN": {f"{a:+g}": delta_vs_control(
                df, "refusal", 44, a, ["KNOWN"]) for a in (3.0, 6.0, 12.0)},
        },
    }

    degen = {cid: {"degeneration_rate": round(float(s["degenerate"].mean()), 4),
                   "empty": int(s["empty"].sum()),
                   "repetitive": int(s["repetitive"].sum()),
                   "non_polish": int(s["non_polish"].sum()),
                   "token_salad": int(s["token_salad"].sum()),
                   "mean_words": round(float(s["n_words"].mean()), 1)}
             for cid, s in df.groupby("config_id")}

    examples = {}
    for cid in ("baseline_alpha0", "familiarity_L30_a-8", "familiarity_L30_a-24",
                "familiarity_L30_a+8", "familiarity_L30_a+24",
                "refusal_L44_a+6", "refusal_L44_a+12", "refusal_L44_a-6",
                "random0_L30_a+24", "random0_L44_a+12"):
        if cid in set(df["config_id"]):
            examples[cid] = pick_examples(df, cid)

    confirm_block = None
    if confirm is not None:
        confirm_block = {}
        for cid, sub in confirm.groupby("config_id"):
            confirm_block[cid] = {
                cond: {"n": len(s),
                       "mean_refusal_rate": round(float(s["refusal_rate"].mean()), 4),
                       "all_refusal": int(s["all_refusal"].sum()),
                       "all_refusal_rate": round(float(s["all_refusal"].mean()), 4),
                       "any_refusal_rate": round(float((s["refusal_rate"] > 0).mean()), 4)}
                for cond, s in sub.groupby("condition")}

    report = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "google/gemma-4-12b-it",
            "n_generations": int(len(df)),
            "refusal_metric": "config.REFUSAL_MARKERS substring match "
                              "(identical to gemma_control.md / P0.1)",
            "decoding": "greedy, max 64 new tokens, pipeline chat template",
            "seed": SEED,
        },
        "directions_meta": json.loads(DIR_META.read_text()) if DIR_META.exists() else None,
        "baseline_reproduction": baseline_block,
        "per_config": per_config,
        "curves": curves(df),
        "hypotheses": hyp,
        "verdicts": verdicts(),
        "degeneration": degen,
        "random_control_coherent_subset": random_coherent_subset(df),
        "haiku_audit": audit_agreement(audit),
        "examples": examples,
        "confirmation": confirm_block,
        "limitations": [
            "Greedy single-sample primary metric (confirmation configs rerun "
            "with 5 samples at T=0.7 to rule out greedy artifacts).",
            "Refusal measured by Polish marker-list substring match; validated "
            "on a 60-generation claude-haiku-4-5 audit, but steering-induced "
            "phrasing outside the marker list would be missed.",
            "One model (Gemma-4-12B), one domain (athletes, 126 entities); "
            "n=42 per condition gives wide Wilson intervals.",
            "Directions are diff-of-means estimates from n=126 prompt-point "
            "activations; single-layer intervention only.",
            "Random-direction controls run at four amplitudes at L30 "
            "(2/4/8/24) and three at L44 (3/6/12), not the full dose curve; "
            "all random controls at |alpha|>=6 degenerate, so the "
            "specificity comparison at high dose rests on the degeneration "
            "contrast plus the low-dose non-degenerate random controls.",
            "The refusal direction is estimated from the SAME entities it is "
            "tested on (no held-out entity split at n=52/32) - causal effect "
            "size may be optimistic relative to unseen entities.",
        ],
    }
    return report


def render_md(rep: dict) -> str:
    L = []
    L.append("# Paper-2 Phase 3 — causal steering of Gemma-4-12B (RQ3)")
    L.append("")
    m = rep["meta"]
    L.append(f"*Generated {m['generated_at']} by scripts/analyze_steering.py from "
             f"{m['n_generations']} greedy generations (scripts/steering_gemma.py). "
             f"Refusal metric: {m['refusal_metric']}. Decoding: {m['decoding']}.*")
    L.append("")

    dm = rep.get("directions_meta") or {}
    L.append("## Directions")
    L.append("")
    L.append(f"- Layer indexing: {dm.get('layer_indexing', 'n/a')}")
    sc = dm.get("sign_convention", {})
    for k, v in sc.items():
        L.append(f"- {k}: {v}")
    L.append("- Cosines: " + ", ".join(f"{k}={v:+.3f}" for k, v in
                                       dm.get("cosines", {}).items()))
    L.append("- Residual norms: " + "; ".join(
        f"L{k}: ||h||={v['mean_l2_norm']:.0f}, rms={v['mean_rms']:.2f}"
        for k, v in dm.get("layers", {}).items()))
    L.append("")

    L.append("## Baseline reproduction (greedy alpha=0 vs stored 5-sample labels)")
    L.append("")
    b = rep["baseline_reproduction"]
    L.append("| Condition | greedy refusal | stored all-refusal |")
    L.append("|---|---|---|")
    for cond, c in b["per_condition"].items():
        g = c["greedy_refusal"]
        L.append(f"| {cond} | {g['refused']}/{g['n']} = {g['rate']:.3f} "
                 f"[{g['wilson95'][0]:.3f}, {g['wilson95'][1]:.3f}] | "
                 f"{c['stored_all_refusal']}/{g['n']} = {c['stored_all_refusal_rate']:.3f} |")
    L.append("")
    L.append(f"Per-entity agreement (greedy vs stored all-refusal): "
             f"**{b['per_entity_agreement_greedy_vs_stored']:.3f}**")
    L.append("")

    L.append("## Refusal-rate curves (rate [Wilson 95%], per condition)")
    for fam_key, conds in rep["curves"].items():
        L.append("")
        L.append(f"### {fam_key}")
        L.append("")
        alphas = sorted({a for c in conds.values() for a in c},
                        key=lambda s: (float(s.split("/")[0]), s))
        L.append("| Condition | " + " | ".join(f"a={a}" for a in alphas) + " |")
        L.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
        for cond, pts in conds.items():
            cells = []
            for a in alphas:
                c = pts.get(a)
                cells.append("—" if c is None else
                             f"{c['rate']:.2f} [{c['wilson95'][0]:.2f},{c['wilson95'][1]:.2f}]")
            L.append(f"| {cond} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Hypotheses")
    for hname, h in rep["hypotheses"].items():
        L.append("")
        L.append(f"### {hname}")
        L.append("")
        L.append(f"Test: {h['test']}")
        for key in (k for k in h if k.endswith("_cells") or k == "cells"
                    or k.startswith("refusal_dir") or k.startswith("familiarity_dir")):
            val = h[key]
            if not isinstance(val, dict):
                continue
            L.append("")
            L.append(f"**{key}**")
            for a, cell in val.items():
                if not isinstance(cell, dict) or "config" not in cell:
                    continue
                c, bb = cell["config"], cell["baseline"]
                line = (f"- alpha {a}: {c['rate']:.3f} [{c['wilson95'][0]:.3f},"
                        f"{c['wilson95'][1]:.3f}] vs baseline {bb['rate']:.3f} "
                        f"(delta {cell['delta_vs_baseline']:+.3f}")
                if "control_delta_vs_baseline" in cell:
                    line += f"; random-control delta {cell['control_delta_vs_baseline']:+.3f}"
                L.append(line + f"; degen {c['degeneration_rate']:.2f})")
        for key in (k for k in h if k.startswith("monotonicity")):
            mo = h[key]
            if mo:
                L.append(f"- {key}: rho={mo['spearman_rho']:+.3f} (p={mo['p']:.3f}) "
                         f"over alphas {mo['alphas']} -> rates {mo['rates']}")
        if "confabulation_rate_urfab" in h:
            L.append(f"- confabulation proxy (coherent non-refusal answers, "
                     f"UR+FAB) per alpha: {h['confabulation_rate_urfab']}")
    L.append("")

    L.append("## Verdicts")
    for hname, text in rep["verdicts"].items():
        L.append("")
        L.append(f"**{hname}** — {text}")
    L.append("")

    L.append("## Degeneration guard (per config)")
    L.append("")
    L.append("| Config | degen rate | empty | repetitive | non-Polish | token salad | mean words |")
    L.append("|---|---|---|---|---|---|---|")
    for cid, d in sorted(rep["degeneration"].items()):
        L.append(f"| {cid} | {d['degeneration_rate']:.3f} | {d['empty']} | "
                 f"{d['repetitive']} | {d['non_polish']} | {d['token_salad']} | "
                 f"{d['mean_words']} |")
    L.append("")

    rcs = rep["random_control_coherent_subset"]
    L.append("## Random controls, coherent-subset refusal rates")
    L.append("")
    L.append("Interpretation guard for refusal-DECREASE claims: random "
             "perturbations also zero the Polish refusal markers, but they do "
             "it by corrupting the output (even the heuristically coherent "
             "subset is visibly damaged). A refusal drop counts as steering "
             "only when outputs stay coherent — which holds for the trained "
             "directions (0% degeneration) and for no random control.")
    L.append("")
    L.append(f"Baseline refusal rate: {rcs['baseline_refusal_rate']:.3f}")
    L.append("")
    L.append("| Random config | coherent n/126 | refusal in coherent subset | baseline expected (mix-adjusted) |")
    L.append("|---|---|---|---|")
    for cid, c in sorted(rcs.items()):
        if not isinstance(c, dict):
            continue
        if "refusal_rate" in c:
            L.append(f"| {cid} | {c['coherent_n']} | {c['refusal_rate']:.3f} "
                     f"[{c['wilson95'][0]:.3f},{c['wilson95'][1]:.3f}] | "
                     f"{c['baseline_expected_for_mix']:.3f} |")
        else:
            L.append(f"| {cid} | {c['coherent_n']} | (too few coherent) | — |")
    L.append("")

    a = rep["haiku_audit"]
    L.append("## Haiku audit of the marker metric")
    L.append("")
    if a.get("n"):
        L.append(f"n={a['n']} random steered generations, {AUDIT_MODEL}; "
                 f"categories {a['haiku_categories']}; binary agreement "
                 f"(haiku refusal vs marker) **{a['binary_agreement']:.3f}** "
                 f"({a['n_disagreements']} disagreements).")
        for d in a.get("disagreements", [])[:8]:
            L.append(f"- [{d['config_id']}] {d['entity']}: marker={d['marker_refusal']}, "
                     f"haiku={d['haiku']} — \"{d['text']}\"")
    else:
        L.append("(audit not run)")
    L.append("")

    L.append("## Example generations")
    for cid, exs in rep["examples"].items():
        L.append("")
        L.append(f"### {cid}")
        for e in exs:
            L.append(f"- ({e['condition']}, {'REFUSAL' if e['is_refusal'] else 'answer'}) "
                     f"**{e['entity']}**: \"{e['text']}\"")
    L.append("")

    if rep.get("confirmation"):
        L.append("## Confirmation runs (5 samples, T=0.7)")
        L.append("")
        for cid, conds in rep["confirmation"].items():
            L.append(f"### {cid}")
            L.append("")
            L.append("| Condition | n | mean refusal_rate | all-refusal | any-refusal |")
            L.append("|---|---|---|---|---|")
            for cond, c in conds.items():
                L.append(f"| {cond} | {c['n']} | {c['mean_refusal_rate']:.3f} | "
                         f"{c['all_refusal']}/{c['n']} = {c['all_refusal_rate']:.3f} | "
                         f"{c['any_refusal_rate']:.3f} |")
            L.append("")

    L.append("## Limitations")
    L.append("")
    for lim in rep["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--n-audit", type=int, default=60)
    args = ap.parse_args()
    df = load_generations()
    print(f"loaded {len(df)} generations across {df['config_id'].nunique()} configs")
    audit = [] if args.no_audit else run_audit(df, args.n_audit)
    confirm = load_confirm()
    rep = build_report(df, audit, confirm)
    REPORT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(render_md(rep))
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
