"""Paper-2 final phase, step 4: RQ4 selective answering (risk-coverage / AURC).

Evaluates every gate score as a selective-answering policy against the
step-2 behavioral correctness labels: the model answers only when the gate's
risk is below a threshold; sweeping the threshold traces the risk-coverage
curve. Target label ``entity_incorrect`` (from p2_calibration.py):
real entity -> majority of 5 strict-judged answers incorrect; fabricated
entity -> majority of 5 answers are confident (non-refusal) confabulations.

Gates (risk orientation: higher = riskier):
  * probe_calibrated       our familiarity probe (Platt-calibrated P(risk); 1 pass)
  * dispersion_best_layer  our best-layer dispersion signal (1 pass)
  * first_token_entropy    zero-cost baseline (1 pass)
  * semantic_entropy       5 sampled answers + judge clustering (5 passes + API)
  * d2hscore               D2HScore risk (1 full generation w/ internals)
  * eigentrack / mind      supervised baselines; for RQ4 they are retrained
                           with 5-fold CV directly on entity_incorrect over
                           the full pool (out-of-fold gate scores)

Metrics: AURC (mean selective error over all coverage prefixes), selective
error at 50% / 80% coverage, full-coverage error; ECE (10 bins) of the
calibrated probe against entity_incorrect.

Outputs: results/paper2_final/{rq4.json, risk_coverage.parquet}.

Usage:
    uv run python scripts/p2_rq4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p2_baselines import SLUGS, gru_cv_probs, load_captures, mlp_cv_probs  # noqa: E402

OUT_ROOT = ROOT / "results" / "paper2_final"

GATES = (
    "probe_calibrated", "dispersion_best_layer", "first_token_entropy",
    "semantic_entropy", "d2hscore", "eigentrack", "mind",
)
GATE_COLUMNS = {
    "probe_calibrated": "risk_probe",
    "dispersion_best_layer": "risk_dispersion",
    "first_token_entropy": "risk_fte",
    "semantic_entropy": "risk_se",
    "d2hscore": "risk_d2h",
    "eigentrack": "rq4_eigentrack_p",
    "mind": "rq4_mind_p",
}
# Inference cost per query, relative to a plain 1-generation deployment.
GATE_COST = {
    "probe_calibrated": "1 forward pass (prompt only, pre-generation)",
    "dispersion_best_layer": "1 forward pass (prompt only, pre-generation)",
    "first_token_entropy": "1 forward pass (prompt only, pre-generation)",
    "semantic_entropy": "5 sampled generations + 1 LLM API clustering call",
    "d2hscore": "1 full generation with hidden-state + attention capture",
    "eigentrack": "1 full generation with hidden-state capture + GRU",
    "mind": "1 full generation with hidden-state capture + MLP",
}


def risk_coverage(risk: np.ndarray, incorrect: np.ndarray) -> dict:
    """Risk-coverage curve + AURC for one gate.

    Entities sorted by ascending risk; coverage k/n answers the k lowest-risk
    entities; selective error = mean(incorrect) among answered. AURC = mean
    selective error over all n prefix coverages (standard discrete AURC).
    """
    mask = np.isfinite(risk)
    risk, incorrect = risk[mask], incorrect[mask].astype(float)
    n = len(risk)
    if n == 0:
        return {"aurc": float("nan")}
    order = np.argsort(risk, kind="stable")
    err_sorted = incorrect[order]
    cum_err = np.cumsum(err_sorted) / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / n
    aurc = float(cum_err.mean())

    def err_at(cov: float) -> float:
        k = max(1, int(round(cov * n)))
        return float(cum_err[k - 1])

    return {
        "aurc": round(aurc, 4),
        "err_at_cov50": round(err_at(0.5), 4),
        "err_at_cov80": round(err_at(0.8), 4),
        "err_full": round(float(incorrect.mean()), 4),
        "n": int(n),
        "coverage": coverages.round(4).tolist(),
        "selective_error": cum_err.round(4).tolist(),
    }


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Expected calibration error with equal-width bins."""
    mask = np.isfinite(probs)
    probs, labels = probs[mask], labels[mask].astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, e = len(probs), 0.0
    reliability = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not sel.any():
            reliability.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0})
            continue
        conf = float(probs[sel].mean())
        acc = float(labels[sel].mean())
        e += sel.sum() / total * abs(conf - acc)
        reliability.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(sel.sum()),
                            "mean_pred_risk": round(conf, 4),
                            "empirical_risk": round(acc, 4)})
    return {"ece": round(float(e), 4), "reliability": reliability}


def main() -> None:
    scores = pd.read_parquet(OUT_ROOT / "baseline_scores.parquet")
    beh = pd.read_parquet(OUT_ROOT / "behavioral_labels.parquet")
    beh_idx = beh.set_index(["slug", "uid"])["entity_incorrect"]

    out: dict = {"per_model": {}, "gate_cost": GATE_COST}
    curve_rows = []

    for slug in SLUGS:
        df = scores[scores["slug"] == slug].reset_index(drop=True)
        if df.empty:
            continue
        y = beh_idx.reindex(list(zip(df["slug"], df["uid"]))).to_numpy().astype(float)
        if np.isnan(y).any():
            raise ValueError(f"{slug}: missing behavioral labels for "
                             f"{int(np.isnan(y).sum())} entities")

        # Supervised gates retrained on the RQ4 target (out-of-fold).
        caps = load_captures(slug)
        uids = df["uid"].tolist()
        seqs = [caps[u]["eig_seq"] for u in uids]
        mindX = np.stack([caps[u]["mind_last"] for u in uids])
        print(f"[rq4] {slug}: retraining EigenTrack/MIND on entity_incorrect",
              flush=True)
        df["rq4_eigentrack_p"] = gru_cv_probs(seqs, y.astype(int))
        df["rq4_mind_p"] = mlp_cv_probs(mindX, y.astype(int))

        model_out: dict = {"n": int(len(df)), "base_error": round(float(y.mean()), 4),
                           "gates": {}}
        for gate in GATES:
            col = GATE_COLUMNS[gate]
            rc = risk_coverage(df[col].to_numpy(dtype=float), y)
            curve = {"coverage": rc.pop("coverage", []),
                     "selective_error": rc.pop("selective_error", [])}
            model_out["gates"][gate] = rc
            for c, e_ in zip(curve["coverage"], curve["selective_error"]):
                curve_rows.append({"slug": slug, "gate": gate,
                                   "coverage": c, "selective_error": e_})
        model_out["probe_ece"] = ece(df["risk_probe"].to_numpy(dtype=float), y)
        out["per_model"][slug] = model_out
        ranked = sorted(model_out["gates"].items(), key=lambda kv: kv[1]["aurc"])
        print(f"[rq4] {slug} AURC ranking: "
              f"{[(g, v['aurc']) for g, v in ranked]}", flush=True)

    # Macro ranking across models.
    gate_means = {}
    for gate in GATES:
        vals = [out["per_model"][s]["gates"][gate]["aurc"]
                for s in out["per_model"]
                if np.isfinite(out["per_model"][s]["gates"][gate]["aurc"])]
        gate_means[gate] = round(float(np.mean(vals)), 4) if vals else float("nan")
    out["macro_aurc"] = dict(sorted(gate_means.items(), key=lambda kv: kv[1]))

    pd.DataFrame(curve_rows).to_parquet(OUT_ROOT / "risk_coverage.parquet")
    with (OUT_ROOT / "rq4.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[rq4] wrote rq4.json + risk_coverage.parquet; "
          f"macro AURC: {out['macro_aurc']}", flush=True)


if __name__ == "__main__":
    main()
