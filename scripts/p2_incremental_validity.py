"""Incremental validity of familiarity.

The behavioral "mirror" may
arise because both familiarity and correctness share popularity as a common
cause. Two questions:

  (1) Does the familiarity score add predictive value for behavioral error
      beyond log-pageviews and domain? (nested logistic + likelihood-ratio test)
  (2) Does familiarity beat a popularity-only gate? (AURC comparison)

Real entities only, per model. Behavioral target = majority of *answered*
samples incorrect (refusals removed, from the refusal-separation reconstruction).

Outputs results/paper2_robustness/incremental_validity.{json,md}.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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


def is_refusal(t): tl = t.lower(); return any(m in tl for m in MARK)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); sr = s[order]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def aurc(risk, incorrect):
    order = np.argsort(risk, kind="mergesort")
    err = incorrect[order].astype(float)
    return float((np.cumsum(err) / (np.arange(len(err)) + 1)).mean())


def answered_incorrect_labels():
    """entity-level: majority of answered samples incorrect (real entities)."""
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
                n_ans = n_ans_corr = 0
                for si, ans in enumerate(d["answers"]):
                    if is_refusal(ans):
                        continue
                    n_ans += 1
                    n_ans_corr += int(bool(judge.get((slug, d["uid"], si), False)))
                rows.append({"slug": slug, "uid": d["uid"], "n_ans": n_ans,
                             "incorrect": (n_ans_corr / n_ans < 0.5) if n_ans else None})
    return pd.DataFrame(rows)


def main():
    lab = answered_incorrect_labels()
    ev = pd.read_parquet(FINAL / "eval_subset.parquet")[
        ["uid", "domain", "pageviews_12m", "kind"]]
    scores = pd.read_parquet(FINAL / "baseline_scores.parquet")[
        ["uid", "slug", "risk_probe"]]

    out = {}
    md = ["# Incremental validity of familiarity beyond popularity (paper 2 rev)\n",
          "Real entities, answered items only. familiarity = 1 - risk_probe; "
          "popularity = log10(pageviews+1); domain dummies.\n",
          "| model | n | base logpv+domain AUROC | +familiarity AUROC | "
          "LR-test p | popularity-only gate AURC | probe gate AURC |",
          "|---|---|---|---|---|---|---|"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for slug in MODELS:
        d = (lab[lab.slug == slug].merge(ev, on="uid", how="left")
             .merge(scores[scores.slug == slug], on="uid", how="left"))
        d = d[d["n_ans"] > 0].dropna(subset=["incorrect", "risk_probe",
                                             "pageviews_12m"])
        y = d["incorrect"].astype(int).to_numpy()
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            md.append(f"| {SHORT[slug]} | {len(y)} | (too imbalanced: "
                      f"{int(y.sum())} incorrect) | | | | |")
            out[SHORT[slug]] = {"n": len(y), "n_incorrect": int(y.sum()),
                                "note": "too imbalanced for a reliable fit"}
            continue
        logpv = np.log10(d["pageviews_12m"].to_numpy() + 1.0)
        dom = pd.get_dummies(d["domain"], drop_first=True).to_numpy(float)
        fam = (1.0 - d["risk_probe"].to_numpy())
        X0 = np.column_stack([logpv, dom])
        X1 = np.column_stack([logpv, dom, fam])

        # likelihood-ratio test (in-sample full-data logistic)
        def loglik(X):
            m = LogisticRegression(max_iter=5000, C=1e6).fit(X, y)
            p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        lr_stat = 2 * (loglik(X1) - loglik(X0))
        lr_p = float(stats.chi2.sf(max(lr_stat, 0.0), df=1))

        # out-of-fold AUROC of each nested model
        a0 = auroc(y, cross_val_predict(LogisticRegression(max_iter=5000),
                   X0, y, cv=skf, method="predict_proba")[:, 1])
        a1 = auroc(y, cross_val_predict(LogisticRegression(max_iter=5000),
                   X1, y, cv=skf, method="predict_proba")[:, 1])
        # gates: popularity-only (risk = -logpv) vs probe risk
        aurc_pop = aurc(-logpv, y)
        aurc_probe = aurc(d["risk_probe"].to_numpy(), y)
        out[SHORT[slug]] = {"n": len(y), "n_incorrect": int(y.sum()),
                            "auroc_pop_domain": round(a0, 3),
                            "auroc_plus_familiarity": round(a1, 3),
                            "lr_test_p": round(lr_p, 4),
                            "aurc_popularity_only": round(aurc_pop, 3),
                            "aurc_probe": round(aurc_probe, 3)}
        md.append(f"| {SHORT[slug]} | {len(y)} | {a0:.3f} | {a1:.3f} | "
                  f"{lr_p:.4f} | {aurc_pop:.3f} | {aurc_probe:.3f} |")

    (OUT / "incremental_validity.json").write_text(json.dumps(out, indent=2))
    (OUT / "incremental_validity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
