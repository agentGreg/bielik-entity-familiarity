"""Direction geometry and statistical hardening.

B2 (shallow version). Cosine geometry of the saved Gemma-12B
steering directions (results/paper2_steering/directions.npz, dim 3840, all
unit-norm): familiarity (L30/L34) vs refusal (L44/L46) diff-of-means, and
familiarity diff-of-means vs the residual-space probe direction saved at the
SAME layer (probe_L30 / probe_L34 = unit-normalized w/sigma of the repo's
StandardScaler+LogisticRegression probe; see scripts/steering_gemma.py).
Calibration: analytic random-cosine scale 1/sqrt(3840) plus an empirical
random-unit-pair distribution.

B3:
  * Paired bootstrap (1000 resamples, seed 0, the SAME resampled entities for
    both methods in a pair) of AUROC differences for the per-cell claims of
    Tables 4-5, from per-entity scores in
    results/paper2_final/baseline_scores.parquet.
    Contrast (a): fabricated-vs-real, n=320 per model (label_fab).
    Contrast (b): behavioral majority-incorrect-vs-correct, REAL entities only,
    n=240 per model (label_beh; EigenTrack/MIND use their contrast-(b)
    retrained out-of-fold scores, per scripts/p2_baselines.py).
    Comparisons per model x contrast: probe vs every other method, plus
    best-vs-runner-up. Point AUROCs are verified against
    results/paper2_final/baselines_auroc.json before bootstrapping.
  * Constant-predictor ECE baseline: ECE with 10 equal-width bins
    (identical binning to the ece() in scripts/p2_rq4.py) of a predictor that
    always outputs the per-model behavioral base rate, against the same
    entity_incorrect target (n=320) as the paper's recalibrated probe ECE.

CPU-only; reads saved artifacts, loads no language model.
Output: results/paper2_robustness/stats_hardening.json.

Usage:
    uv run python scripts/p2_stats_hardening.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "paper2_final"
STEER = ROOT / "results" / "paper2_steering"
OUT_DIR = ROOT / "results" / "paper2_robustness"

SEED = 0
N_BOOT = 1000
D_MODEL = 3840  # Gemma-12B residual stream

SLUGS = (
    "Bielik-11B-v3.0-Instruct",
    "gemma-4-12b-it",
    "Qwen3-14B",
    "PLLuM-12B-instruct-2512",
)

# Method -> per-entity score column, per contrast (mirrors p2_baselines.py).
METHODS_A = {
    "d2hscore": "risk_d2h",
    "eigentrack": "eigentrack_p_fab",
    "mind": "mind_p_fab",
    "dispersion_best_layer": "risk_dispersion",
    "probe_calibrated": "risk_probe",
    "first_token_entropy": "risk_fte",
    "semantic_entropy": "risk_se",
}
METHODS_B = {
    "d2hscore": "risk_d2h",
    "eigentrack": "eigentrack_p_beh",
    "mind": "mind_p_beh",
    "dispersion_best_layer": "risk_dispersion",
    "probe_calibrated": "risk_probe",
    "first_token_entropy": "risk_fte",
    "semantic_entropy": "risk_se",
}


# ---------------------------------------------------------------------------
# B2: direction geometry
# ---------------------------------------------------------------------------

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def direction_geometry() -> dict:
    npz = np.load(STEER / "directions.npz")
    d = {k: npz[k].astype(np.float64) for k in npz.files}
    norms = {k: round(float(np.linalg.norm(v)), 6) for k, v in d.items()}

    cosines = {
        "familiarity_L30_vs_refusal_L44": cos(d["familiarity_L30"], d["refusal_L44"]),
        "familiarity_L34_vs_refusal_L44": cos(d["familiarity_L34"], d["refusal_L44"]),
        "familiarity_L30_vs_refusal_L46": cos(d["familiarity_L30"], d["refusal_L46"]),
        "familiarity_L34_vs_refusal_L46": cos(d["familiarity_L34"], d["refusal_L46"]),
        "familiarity_L30_vs_probe_L30_residual_space": cos(
            d["familiarity_L30"], d["probe_L30"]),
        "familiarity_L34_vs_probe_L34_residual_space": cos(
            d["familiarity_L34"], d["probe_L34"]),
        "familiarity_L30_vs_L34": cos(d["familiarity_L30"], d["familiarity_L34"]),
        "refusal_L44_vs_L46": cos(d["refusal_L44"], d["refusal_L46"]),
    }

    # Random-unit-vector calibration in R^3840.
    rng = np.random.default_rng(SEED)
    rc = np.array([
        cos(rng.standard_normal(D_MODEL), rng.standard_normal(D_MODEL))
        for _ in range(2000)
    ])
    calibration = {
        "analytic_sd_random_cosine_1_over_sqrt_d": 1.0 / np.sqrt(D_MODEL),
        "empirical_mean_abs_cosine": float(np.abs(rc).mean()),
        "empirical_p95_abs_cosine": float(np.percentile(np.abs(rc), 95)),
        "empirical_max_abs_cosine": float(np.abs(rc).max()),
        "n_random_pairs": len(rc),
    }

    # Standardized-space / raw-w probe cosines are only stored in the meta
    # (the npz keeps the residual-space probe direction w/sigma); echo them.
    meta = json.loads((STEER / "directions_meta.json").read_text())
    return {
        "source": str((STEER / "directions.npz").relative_to(ROOT)),
        "dim": D_MODEL,
        "norms": norms,
        "cosines": {k: round(v, 6) for k, v in cosines.items()},
        "probe_cosines_from_meta": {
            k: v for k, v in meta["cosines"].items() if "probe" in k},
        "random_cosine_calibration": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in calibration.items()},
        "note": ("All directions unit-norm in float32. probe_L{30,34} are the "
                 "residual-space gradients (w/sigma, unit norm) of the K-vs-F "
                 "steering-layer probes; the Table-4 probe for Gemma uses "
                 "layer 17 and its weight vector is not persisted "
                 "(train_risk_probe.py saves risk_probe.npz for Bielik slugs "
                 "only). Per-layer familiarity directions exist only at "
                 "L30/L34."),
    }


# ---------------------------------------------------------------------------
# B3.1-2: paired bootstrap of AUROC differences
# ---------------------------------------------------------------------------

def paired_bootstrap(s1: np.ndarray, s2: np.ndarray, y: np.ndarray) -> dict:
    """Paired bootstrap of AUROC(s1) - AUROC(s2) over the same entities."""
    mask = np.isfinite(s1) & np.isfinite(s2)
    s1, s2, y = s1[mask], s2[mask], y[mask].astype(int)
    a1 = roc_auc_score(y, s1)
    a2 = roc_auc_score(y, s2)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue  # degenerate resample (same convention as p2_baselines)
        diffs.append(roc_auc_score(y[idx], s1[idx])
                     - roc_auc_score(y[idx], s2[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auroc_1": round(float(a1), 4),
        "auroc_2": round(float(a2), 4),
        "diff": round(float(a1 - a2), 4),
        "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_boot_used": len(diffs),
    }


def bootstrap_tables(scores: pd.DataFrame, ref: dict) -> dict:
    out: dict = {}
    for slug in SLUGS:
        df = scores[scores["slug"] == slug].reset_index(drop=True)
        out[slug] = {}
        for contrast, methods, ref_key in (
            ("contrast_a_fab_vs_real", METHODS_A, "contrast_a_fab_vs_real"),
            ("contrast_b_behavioral_real_only", METHODS_B,
             "contrast_b_behavioral_real_only"),
        ):
            if contrast.startswith("contrast_b"):
                sub = df[df["kind"] == "real"]
                y = sub["label_beh"].to_numpy()
            else:
                sub = df
                y = sub["label_fab"].to_numpy()

            # Point AUROCs + verification against baselines_auroc.json.
            point, mismatches = {}, []
            for name, col in methods.items():
                s = sub[col].to_numpy(dtype=np.float64)
                m = np.isfinite(s)
                a = round(float(roc_auc_score(y[m].astype(int), s[m])), 4)
                point[name] = a
                stored = ref["per_model"][slug][ref_key][name]["auroc"]
                if abs(a - stored) > 5e-4:
                    mismatches.append((name, a, stored))
            if mismatches:
                raise RuntimeError(
                    f"{slug} {contrast}: recomputed AUROC does not match "
                    f"baselines_auroc.json: {mismatches}")

            ranked = sorted(point, key=point.get, reverse=True)
            comparisons = {}
            # Probe vs every other method.
            for name in methods:
                if name == "probe_calibrated":
                    continue
                comparisons[f"probe_calibrated_vs_{name}"] = paired_bootstrap(
                    sub["risk_probe"].to_numpy(dtype=np.float64),
                    sub[methods[name]].to_numpy(dtype=np.float64), y)
            # Best vs runner-up (the bolded-cell claim).
            best, runner = ranked[0], ranked[1]
            comparisons["best_vs_runner_up"] = {
                "best": best, "runner_up": runner,
                **paired_bootstrap(
                    sub[methods[best]].to_numpy(dtype=np.float64),
                    sub[methods[runner]].to_numpy(dtype=np.float64), y),
            }
            # Probe vs best non-probe (headline claim for contrast a).
            best_np = next(n for n in ranked if n != "probe_calibrated")
            comparisons["probe_vs_best_nonprobe"] = {
                "best_nonprobe": best_np,
                **paired_bootstrap(
                    sub["risk_probe"].to_numpy(dtype=np.float64),
                    sub[methods[best_np]].to_numpy(dtype=np.float64), y),
            }
            out[slug][contrast] = {
                "point_auroc": point,
                "ranking": ranked,
                "comparisons": comparisons,
            }
    return out


# ---------------------------------------------------------------------------
# B3.3: constant-predictor ECE baseline
# ---------------------------------------------------------------------------

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins (verbatim p2_rq4 binning)."""
    mask = np.isfinite(probs)
    probs, labels = probs[mask], labels[mask].astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, e = len(probs), 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not sel.any():
            continue
        e += sel.sum() / total * abs(float(probs[sel].mean())
                                     - float(labels[sel].mean()))
    return float(e)


def constant_ece(scores: pd.DataFrame) -> dict:
    beh = pd.read_parquet(FINAL / "behavioral_labels.parquet")
    beh_idx = beh.set_index(["slug", "uid"])["entity_incorrect"]
    rq4 = json.loads((FINAL / "rq4.json").read_text())
    out = {}
    for slug in SLUGS:
        df = scores[scores["slug"] == slug]
        y = beh_idx.reindex(list(zip(df["slug"], df["uid"]))).to_numpy()
        y = y.astype(float)
        if np.isnan(y).any():
            raise ValueError(f"{slug}: missing behavioral labels")
        base = float(y.mean())
        const = np.full(len(y), base)
        out[slug] = {
            "n": int(len(y)),
            "behavioral_base_rate_incorrect": round(base, 4),
            "constant_predictor_ece_10bins": round(ece(const, y), 4),
            "probe_ece_raw": rq4["per_model"][slug]["probe_ece"]["ece"],
            "probe_ece_recalibrated":
                rq4["per_model"][slug]["probe_ece_recalibrated"]["ece"],
        }
    return out


def main() -> None:
    scores = pd.read_parquet(FINAL / "baseline_scores.parquet")
    ref = json.loads((FINAL / "baselines_auroc.json").read_text())

    result = {
        "meta": {
            "script": "scripts/p2_stats_hardening.py",
            "seed": SEED,
            "n_boot": N_BOOT,
            "bootstrap": ("paired: identical entity resamples for both methods; "
                          "resamples with a single class skipped"),
            "inputs": [
                "results/paper2_steering/directions.npz",
                "results/paper2_final/baseline_scores.parquet",
                "results/paper2_final/behavioral_labels.parquet",
                "results/paper2_final/baselines_auroc.json",
                "results/paper2_final/rq4.json",
            ],
        },
        "b2_direction_geometry": direction_geometry(),
        "b3_paired_bootstrap": bootstrap_tables(scores, ref),
        "b3_constant_ece": constant_ece(scores),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stats_hardening.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")

    # Console summary.
    g = result["b2_direction_geometry"]["cosines"]
    print("\n[B2] cos(familiarity_L30, refusal_L44) ="
          f" {g['familiarity_L30_vs_refusal_L44']:+.4f}")
    print(f"[B2] cos(familiarity_L30, probe_L30)   ="
          f" {g['familiarity_L30_vs_probe_L30_residual_space']:+.4f}")
    for slug, contrasts in result["b3_paired_bootstrap"].items():
        for contrast, cell in contrasts.items():
            c = cell["comparisons"]["best_vs_runner_up"]
            print(f"[B3] {slug} {contrast}: {c['best']} {c['auroc_1']} vs "
                  f"{c['runner_up']} {c['auroc_2']} diff {c['diff']:+.4f} "
                  f"CI {c['diff_ci95']} excl0={c['ci_excludes_zero']}")
    for slug, e in result["b3_constant_ece"].items():
        print(f"[B3] {slug}: constant-ECE {e['constant_predictor_ece_10bins']} "
              f"(base rate {e['behavioral_base_rate_incorrect']}) vs "
              f"recalibrated probe ECE {e['probe_ece_recalibrated']}")


if __name__ == "__main__":
    main()
