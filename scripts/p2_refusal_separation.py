"""Separate refusal from correctness.

The strict judge scores explicit
refusals as correct ~88% of the time, so every behavioral metric that folds
refusals into "correct" is contaminated for the one abstaining model (Gemma).

This script reconstructs three separate variables from saved artifacts, with NO
new judge calls:

  1. answered vs. abstained   -- per-answer refusal marker (published list;
     reconciles to Table 6's 533/1200 Gemma real refusals)
  2. correctness | answered   -- existing per-sample judge label, restricted to
     answered samples
  3. system utility           -- error rate under three refusal treatments

and recomputes, among ANSWERED items:
  - behavioral AUROC (block b) for every gate, vs. the contaminated version
  - base error (overall vs. answered-only)
  - risk-coverage AURC treating refusals as abstentions removed from coverage

Outputs results/paper2_robustness/refusal_separation.{json,md}.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/paper2_final"
OUT = ROOT / "results/paper2_robustness"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Bielik-11B-v3.0-Instruct", "gemma-4-12b-it",
          "Qwen3-14B", "PLLuM-12B-instruct-2512"]
SHORT = {"Bielik-11B-v3.0-Instruct": "Bielik-11B", "gemma-4-12b-it": "Gemma-12B",
         "Qwen3-14B": "Qwen-14B", "PLLuM-12B-instruct-2512": "PLLuM-12B"}
MARK = ["nie wiem", "nie mam informacji", "nie posiadam",
        "nie jestem w stanie", "nie znam", "brak informacji"]

# gate score columns in baseline_scores.parquet (higher risk_* = more likely wrong)
GATES = {"probe": "risk_probe", "dispersion": "risk_dispersion",
         "FT entropy": "risk_fte", "semantic entropy": "risk_se",
         "MIND": "mind_p_beh", "EigenTrack": "eigentrack_p_beh"}


def is_refusal(text: str) -> bool:
    tl = text.lower()
    return any(m in tl for m in MARK)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC of score s for positive label y (1). Rank-based; ties averaged."""
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def aurc(risk_scores: np.ndarray, incorrect: np.ndarray) -> float:
    """Area under risk-coverage curve. Rank by ascending risk; mean selective
    error over all coverage prefixes. Lower is better."""
    order = np.argsort(risk_scores, kind="mergesort")
    err = incorrect[order].astype(float)
    cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    return float(cum.mean())


def load_per_answer() -> pd.DataFrame:
    """One row per (slug, uid, sample_idx): refused, correct (answered only)."""
    judge = {}
    for line in open(FINAL / "judge_checkpoint.jsonl"):
        d = json.loads(line)
        judge[(d["slug"], d["uid"], d["sample_idx"])] = bool(d["correct"])
    rows = []
    for slug in MODELS:
        for f in glob.glob(str(FINAL / f"gen/{slug}/*/answers.jsonl")):
            for line in open(f):
                d = json.loads(line)
                if d["kind"] != "real":
                    continue
                for si, ans in enumerate(d["answers"]):
                    ref = is_refusal(ans)
                    rows.append({
                        "slug": slug, "uid": d["uid"], "domain": d["domain"],
                        "decile": d["decile"], "sample_idx": si,
                        "refused": ref,
                        "judge_correct": judge.get((slug, d["uid"], si), None),
                    })
    return pd.DataFrame(rows)


def main():
    pa = load_per_answer()
    scores = pd.read_parquet(FINAL / "baseline_scores.parquet")

    summary = {}
    md = ["# Refusal separated from correctness\n",
          "Per-answer refusal via the published marker list; correctness from the "
          "existing per-sample judge labels. No new judge calls.\n"]

    # ---- entity-level clean labels -------------------------------------
    ent = (pa.groupby(["slug", "uid", "domain", "decile"])
             .agg(n=("refused", "size"),
                  n_refused=("refused", "sum"),
                  n_correct_any=("judge_correct", "sum"))  # judge-correct incl. refusals
             .reset_index())
    # correctness conditional on answering
    ans = pa[~pa["refused"]].copy()
    ans_ent = (ans.groupby(["slug", "uid"])
                 .agg(n_ans=("judge_correct", "size"),
                      n_ans_correct=("judge_correct", "sum"))
                 .reset_index())
    ent = ent.merge(ans_ent, on=["slug", "uid"], how="left")
    ent["n_ans"] = ent["n_ans"].fillna(0).astype(int)
    ent["n_ans_correct"] = ent["n_ans_correct"].fillna(0).astype(int)
    ent["abstained_entity"] = ent["n_ans"] == 0
    # behavioral target among answered: majority of answered samples incorrect
    with np.errstate(invalid="ignore"):
        frac_ans_correct = ent["n_ans_correct"] / ent["n_ans"].replace(0, np.nan)
    ent["answered_incorrect_maj"] = (frac_ans_correct < 0.5).astype("Int64")

    # ---- per-model tables ----------------------------------------------
    rows_summary = []
    behav_rows = []
    aurc_rows = []
    for slug in MODELS:
        e = ent[ent.slug == slug]
        n_ent = len(e)
        n_answered_ent = int((~e["abstained_entity"]).sum())
        per_ans = pa[pa.slug == slug]
        n_ans_total = int((~per_ans["refused"]).sum())
        n_ref_total = int(per_ans["refused"].sum())
        # correctness rates
        overall_correct = float(per_ans["judge_correct"].mean())          # contaminated
        answered_correct = float(ans[ans.slug == slug]["judge_correct"].mean()) \
            if n_ans_total else float("nan")
        rows_summary.append({
            "model": SHORT[slug], "entities": n_ent,
            "answers": len(per_ans), "refused_answers": n_ref_total,
            "refusal_rate": round(n_ref_total / len(per_ans), 3),
            "correct_all_incl_refusal": round(overall_correct, 3),
            "correct_given_answered": round(answered_correct, 3),
            "base_err_all": round(1 - overall_correct, 3),
            "base_err_answered": round(1 - answered_correct, 3),
        })

        # behavioral AUROC among answered entities (>=1 answered sample)
        sc = scores[scores.slug == slug].set_index("uid")
        ea = e[~e["abstained_entity"]].copy()
        y = ea["answered_incorrect_maj"].astype(int).to_numpy()
        gate_auroc = {}
        for gname, col in GATES.items():
            s = sc.reindex(ea["uid"])[col].to_numpy(dtype=float)
            m = ~np.isnan(s)
            gate_auroc[gname] = round(auroc(y[m], s[m]), 3)
        behav_rows.append({"model": SHORT[slug],
                           "n_answered_entities": n_answered_ent,
                           "n_incorrect": int(y.sum()), **gate_auroc})

        # AURC treating refusals as abstentions removed from coverage:
        # rank answered entities by gate risk; error = answered_incorrect_maj
        gate_aurc = {}
        for gname, col in GATES.items():
            s = sc.reindex(ea["uid"])[col].to_numpy(dtype=float)
            m = ~np.isnan(s)
            gate_aurc[gname] = round(aurc(s[m], y[m]), 3)
        aurc_rows.append({"model": SHORT[slug],
                          "base_err_answered": round(1 - answered_correct, 3)
                          if n_ans_total else None, **gate_aurc})

    summary["answered_correct_breakdown"] = rows_summary
    summary["behavioral_auroc_answered"] = behav_rows
    summary["aurc_answered"] = aurc_rows

    # ---- markdown ------------------------------------------------------
    def tbl(rows, cols):
        h = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
                for r in rows]
        return "\n".join([h, sep] + body)

    md.append("## Three variables: answered / refused / correct\n")
    md.append(tbl(rows_summary, ["model", "answers", "refused_answers",
              "refusal_rate", "correct_all_incl_refusal",
              "correct_given_answered", "base_err_all", "base_err_answered"]))
    md.append("\n\n## Behavioral AUROC among ANSWERED items (block b, cleaned)\n")
    md.append("Target: majority of *answered* samples incorrect. Refusals removed "
              "(not scored as correct). Compare Gemma's probe cell to the "
              "contaminated 0.409 in the current Table 4b.\n")
    md.append(tbl(behav_rows, ["model", "n_answered_entities", "n_incorrect"]
                  + list(GATES)))
    md.append("\n\n## AURC among answered (refusals = abstention, removed from coverage)\n")
    md.append(tbl(aurc_rows, ["model", "base_err_answered"] + list(GATES)))
    (OUT / "refusal_separation.md").write_text("\n".join(md) + "\n")
    (OUT / "refusal_separation.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(md))
    print(f"\nWrote {OUT/'refusal_separation.md'}")


if __name__ == "__main__":
    main()
