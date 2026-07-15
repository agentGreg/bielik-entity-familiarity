"""Analysis of the paper-2 Phase-2 v2 campaign (popularity gradation).

Runs incrementally: analyses whatever (model, domain) outputs exist under
``results/<slug>/v2/<domain>/`` and writes ``results/v2_campaign.{json,md}``
(the ``analysis`` block of the JSON + a human-readable markdown report).

Per (model, domain) it computes
--------------------------------
1. **Familiarity gradation (Spearman rho)** of two familiarity scores vs
   ``log10(pageviews_12m + 1)`` among REAL entities:
     * ``dispersion`` — the best-layer dispersion metric value, where "best" is
       the (point, metric, layer) with the cleanest anchor contrast, i.e. the
       highest AUROC separating **real top-decile (9) vs fabricated**. Oriented
       so higher = more familiar.
     * ``probe`` — residual-stream logistic-probe score P(real), trained on the
       **top-3 deciles (7,8,9) vs fabricated** at the probe's best layer (same
       top-decile-vs-fabricated layer selection), applied to ALL real entities
       (out-of-fold for the training deciles, direct for the rest).
2. **Per-decile mean score curve** (headline plot data): mean dispersion &
   probe score for deciles 0-9 plus the fabricated anchor (decile -1).
3. **Adjacent-pair separability**: AUROC of each real decile vs fabricated
   (does separability rise monotonically with popularity?).
4. **Cross-family summary**: rho per model ordered by size within family.
5. **PLLuM-vs-base note**: PLLuM-12B (Mistral base) & Llama-PLLuM-8B (Llama
   base) — the Polish continual-pretraining axis.

Bootstrap 95% CIs (2000 resamples) on the headline rhos.

Usage
-----
    .venv/bin/python scripts/analyze_v2.py
    .venv/bin/python scripts/analyze_v2.py --models Bielik-1.5B --domains cities
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from run_v2_campaign import MODELS, DOMAINS, slug_for, results_dir_for, data_dir_for  # noqa: E402

CAMPAIGN_JSON = ROOT / "results" / "v2_campaign.json"
CAMPAIGN_MD = ROOT / "results" / "v2_campaign.md"

# Family grouping for the cross-family scaling table (label -> family).
FAMILY = {
    "Bielik-1.5B": "Bielik", "Bielik-4.5B": "Bielik",
    "Bielik-Minitron-7B": "Bielik", "Bielik-11B": "Bielik",
    "Qwen3-1.7B": "Qwen3", "Qwen3-4B": "Qwen3", "Qwen3-14B": "Qwen3",
    "PLLuM-4B": "PLLuM", "Llama-PLLuM-8B": "PLLuM", "PLLuM-12B": "PLLuM",
    "gemma-4-E4B": "Gemma", "gemma-4-12b": "Gemma",
}
# Approximate parameter count (B) for within-family ordering.
SIZE_B = {
    "Bielik-1.5B": 1.5, "Qwen3-1.7B": 1.7, "PLLuM-4B": 4.3, "Qwen3-4B": 4.0,
    "Bielik-4.5B": 4.5, "Bielik-Minitron-7B": 7.0, "Llama-PLLuM-8B": 8.0,
    "Bielik-11B": 11.0, "gemma-4-E4B": 4.0, "gemma-4-12b": 12.0,
    "PLLuM-12B": 12.25, "Qwen3-14B": 14.0,
}

MetricCols = ("ipr", "entropy")
POINTS = ("entity", "prompt")


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    a = roc_auc_score(labels, scores)
    return float(max(a, 1.0 - a))


def _bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, n: int = 2000,
                      seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(x)
    if m < 3:
        return (float("nan"), float("nan"))
    rhos = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        r = spearmanr(x[idx], y[idx]).statistic
        rhos[i] = r if r == r else 0.0
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return (float(lo), float(hi))


def _load_signals(model_id: str, domain: str):
    rd = results_dir_for(model_id, domain)
    sig_path = rd / "signals.parquet"
    npz_path = rd / "hidden_states.npz"
    lab_path = data_dir_for(model_id, domain) / "labeled.parquet"
    if not (sig_path.exists() and npz_path.exists() and lab_path.exists()):
        return None
    sig = pd.read_parquet(sig_path)
    labeled = pd.read_parquet(lab_path)  # carries decile / pageviews / kind
    return sig, labeled, npz_path


def _dispersion_analysis(sig: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Select best (point, metric, layer) by real-top-decile-9-vs-fabricated
    AUROC, then read that score per REAL entity and correlate vs log pageviews.
    Also compute per-decile mean curves + per-decile-vs-fabricated AUROC."""
    # entity -> decile / kind / logpv, ordered as in meta.
    m = meta.set_index("entity")
    logpv = np.log10(m["pageviews_12m"].to_numpy() + 1.0)

    best = {"auroc": -1.0, "point": None, "metric": None, "layer": None,
            "orient": 1.0}
    # Precompute pivot per (point, metric): entity x layer.
    for point in POINTS:
        sub = sig[sig["point"] == point]
        for metric in MetricCols:
            piv = sub.pivot_table(index="entity", columns="layer",
                                  values=metric, aggfunc="first")
            piv = piv.reindex(m.index)
            kind = m["kind"].to_numpy()
            dec = m["decile"].to_numpy()
            # top decile 9 vs fabricated
            sel = (dec == 9) | (kind == "fabricated")
            labs = (kind[sel] == "fabricated").astype(int)  # fabricated=1 (unfamiliar)
            for layer in piv.columns:
                vals = piv[layer].to_numpy()
                s = vals[sel]
                if np.isnan(s).any() or len(np.unique(labs)) < 2:
                    continue
                a = _auroc(s, labs)
                if a > best["auroc"]:
                    # orient so higher score = more familiar (real). If raw AUROC
                    # (fabricated=1) < 0.5 the metric already rises with real; we
                    # store orient to make the familiarity score increase with
                    # popularity. Determine orientation by sign of mean(real)-mean(fab).
                    fab_mean = np.nanmean(vals[kind == "fabricated"])
                    real_mean = np.nanmean(vals[kind == "real"])
                    orient = 1.0 if real_mean >= fab_mean else -1.0
                    best.update(auroc=a, point=point, metric=metric,
                                layer=int(layer), orient=orient)

    if best["point"] is None:
        return {"ok": False}

    sub = sig[(sig["point"] == best["point"])]
    piv = sub.pivot_table(index="entity", columns="layer",
                          values=best["metric"], aggfunc="first").reindex(m.index)
    score = piv[best["layer"]].to_numpy() * best["orient"]

    kind = m["kind"].to_numpy()
    dec = m["decile"].to_numpy()
    real = kind == "real"

    rho = spearmanr(score[real], logpv[real]).statistic
    lo, hi = _bootstrap_rho_ci(score[real], logpv[real])

    # per-decile mean curve (real 0-9) + fabricated anchor (-1)
    curve = {}
    for d in range(10):
        v = score[real & (dec == d)]
        curve[str(d)] = float(np.nanmean(v)) if len(v) else None
    fab = score[kind == "fabricated"]
    curve["-1"] = float(np.nanmean(fab)) if len(fab) else None

    # per-decile vs fabricated AUROC
    pair = {}
    fab_scores = score[kind == "fabricated"]
    for d in range(10):
        dv = score[real & (dec == d)]
        if len(dv) and len(fab_scores):
            s = np.concatenate([dv, fab_scores])
            labs = np.concatenate([np.ones(len(dv)), np.zeros(len(fab_scores))])
            pair[str(d)] = _auroc(s, labs)
    return {
        "ok": True,
        "best_point": best["point"], "best_metric": best["metric"],
        "best_layer": best["layer"], "anchor_auroc": round(best["auroc"], 4),
        "rho": float(rho), "rho_ci95": [round(lo, 4), round(hi, 4)],
        "decile_curve": curve, "decile_vs_fab_auroc": {k: round(v, 4) for k, v in pair.items()},
    }


def _probe_analysis(npz_path: Path, meta: pd.DataFrame) -> dict:
    """Train a residual-stream probe on top-3-deciles(7,8,9)-vs-fabricated,
    apply to ALL real entities; select the layer by the same top-decile-vs-fab
    AUROC. Score = P(real). Correlate vs log pageviews among real entities."""
    m = meta.set_index("entity")
    entities = m.index.to_numpy()
    kind = m["kind"].to_numpy()
    dec = m["decile"].to_numpy()
    logpv = np.log10(m["pageviews_12m"].to_numpy() + 1.0)

    with np.load(npz_path, allow_pickle=False) as npz:
        layer_keys = sorted(
            (int(k.split("_")[-1]) for k in npz.files if k.startswith("prompt_layer_")))
        H = {L: npz[f"prompt_layer_{L}"] for L in layer_keys}
        # npz rows follow labeled.parquet row order (extract iterates df rows).
    n = len(entities)
    for L, X in H.items():
        if len(X) != n:
            return {"ok": False, "reason": f"layer {L} rows {len(X)} != {n}"}

    train_mask = ((kind == "real") & np.isin(dec, [7, 8, 9])) | (kind == "fabricated")
    y_train = (kind[train_mask] == "real").astype(int)  # real=1 (familiar)
    if len(np.unique(y_train)) < 2:
        return {"ok": False, "reason": "single-class train set"}

    # Select best layer by CV AUROC on the training contrast.
    minority = int(np.bincount(y_train).min())
    n_splits = min(5, minority)
    best = {"auroc": -1.0, "layer": None}
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for L, X in H.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            proba = cross_val_predict(clf, X[train_mask], y_train, cv=cv,
                                      method="predict_proba")[:, 1]
            a = _auroc(proba, y_train)
            if a > best["auroc"]:
                best.update(auroc=a, layer=L)
    if best["layer"] is None:
        return {"ok": False, "reason": "no layer selected"}

    L = best["layer"]
    X = H[L]
    real = kind == "real"
    # Out-of-fold P(real) for training reals; direct predict for non-training reals.
    score = np.full(n, np.nan)
    scaler = StandardScaler().fit(X[train_mask])
    # OOF for the training entities:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        X[train_mask], y_train, cv=cv, method="predict_proba")[:, 1]
    score[train_mask] = oof
    # Fit on all training rows, predict the remaining reals (deciles 0-6):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X[train_mask], y_train)
    rest = real & ~train_mask
    if rest.any():
        score[rest] = clf.predict_proba(X[rest])[:, 1]

    rho = spearmanr(score[real], logpv[real]).statistic
    lo, hi = _bootstrap_rho_ci(score[real], logpv[real])

    curve = {}
    for d in range(10):
        v = score[real & (dec == d)]
        curve[str(d)] = float(np.nanmean(v)) if len(v) and not np.isnan(v).all() else None
    # fabricated anchor: predict P(real) on fabricated (they were in training as label 0)
    fab = kind == "fabricated"
    fab_score = clf.predict_proba(X[fab])[:, 1] if fab.any() else np.array([])
    curve["-1"] = float(np.nanmean(fab_score)) if len(fab_score) else None

    return {
        "ok": True, "best_layer": int(L), "anchor_auroc": round(best["auroc"], 4),
        "rho": float(rho), "rho_ci95": [round(lo, 4), round(hi, 4)],
        "decile_curve": curve,
    }


def analyze_pair(model_id: str, domain: str) -> dict | None:
    loaded = _load_signals(model_id, domain)
    if loaded is None:
        return None
    sig, labeled, npz_path = loaded
    disp = _dispersion_analysis(sig, labeled)
    probe = _probe_analysis(npz_path, labeled)
    return {"dispersion": disp, "probe": probe}


def _monotonicity_verdict(pair_auroc: dict) -> str:
    """Rough monotonicity check on per-decile-vs-fabricated AUROC."""
    xs = [pair_auroc[str(d)] for d in range(10) if str(d) in pair_auroc]
    if len(xs) < 3:
        return "n/a"
    # Spearman of AUROC vs decile index.
    idx = list(range(len(xs)))
    r = spearmanr(idx, xs).statistic
    if r != r:
        return "n/a"
    if r >= 0.5:
        return f"rising (rho={r:.2f})"
    if r <= -0.5:
        return f"falling (rho={r:.2f})"
    return f"flat (rho={r:.2f})"


def build_reports(models, domains) -> dict:
    analysis = {}
    for label, model_id in models:
        for domain in domains:
            res = analyze_pair(model_id, domain)
            if res is None:
                continue
            analysis.setdefault(label, {})[domain] = res

    # Merge into campaign JSON.
    summary = {}
    if CAMPAIGN_JSON.exists():
        try:
            summary = json.loads(CAMPAIGN_JSON.read_text())
        except Exception:
            summary = {}
    summary["analysis"] = analysis
    CAMPAIGN_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    _write_markdown(analysis)
    return analysis


def _fmt_rho(r) -> str:
    if r is None:
        return "—"
    if isinstance(r, dict):
        if not r.get("ok"):
            return "—"
        ci = r.get("rho_ci95", [None, None])
        return f"{r['rho']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
    return f"{r:.3f}"


def _write_markdown(analysis: dict) -> None:
    lines = ["# Paper-2 Phase-2 v2 campaign — popularity gradation\n",
             "Familiarity score vs log10(pageviews+1) among REAL entities. "
             "`rho [lo,hi]` = Spearman with bootstrap 95% CI (2k). "
             "Dispersion score = best-layer metric at the real-top-decile-vs-"
             "fabricated anchor; probe score = P(real) from a top-3-deciles-vs-"
             "fabricated logistic probe.\n"]

    # (4) Cross-family summary table, ordered by size within family.
    lines.append("## Cross-family gradation (rho), ordered by size within family\n")
    lines.append("| Family | Model | ~params (B) | domain | dispersion rho | probe rho | monotonicity (decile-vs-fab AUROC) |")
    lines.append("|---|---|---|---|---|---|---|")
    ordered = sorted(analysis.keys(),
                     key=lambda l: (FAMILY.get(l, "zz"), SIZE_B.get(l, 99)))
    for label in ordered:
        for domain in DOMAINS:
            if domain not in analysis[label]:
                continue
            d = analysis[label][domain]["dispersion"]
            p = analysis[label][domain]["probe"]
            mono = _monotonicity_verdict(d.get("decile_vs_fab_auroc", {})) if d.get("ok") else "—"
            lines.append(
                f"| {FAMILY.get(label,'?')} | {label} | {SIZE_B.get(label,'?')} | "
                f"{domain} | {_fmt_rho(d)} | {_fmt_rho(p)} | {mono} |")

    # (5) PLLuM-vs-base note.
    lines.append("\n## PLLuM continual-pretraining axis (Polish CPT vs base family)\n")
    lines.append("PLLuM-12B is Mistral-Nemo-based; Llama-PLLuM-8B is Llama-3.1-based. "
                 "Compare their gradation rho against the same-architecture reference "
                 "points (Qwen3/Bielik at nearby scale).\n")
    for label in ("Llama-PLLuM-8B", "PLLuM-12B", "PLLuM-4B"):
        if label in analysis:
            rhos = [analysis[label][dm]["dispersion"]["rho"]
                    for dm in analysis[label]
                    if analysis[label][dm]["dispersion"].get("ok")]
            if rhos:
                lines.append(f"- **{label}** ({FAMILY.get(label)}): mean dispersion "
                             f"rho over {len(rhos)} domains = {np.mean(rhos):.3f}")

    # (2/3) Per-decile curves for representative models.
    lines.append("\n## Per-decile mean dispersion curve (representative models)\n")
    reps = [m for m in ("Bielik-1.5B", "Qwen3-4B", "Bielik-11B") if m in analysis]
    for label in reps:
        lines.append(f"\n### {label}\n")
        lines.append("| domain | fab(-1) | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |")
        lines.append("|---|" + "---|" * 11)
        for domain in DOMAINS:
            if domain not in analysis[label]:
                continue
            c = analysis[label][domain]["dispersion"].get("decile_curve", {})
            def g(k):
                v = c.get(k)
                return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
            row = [g("-1")] + [g(str(d)) for d in range(10)]
            lines.append(f"| {domain} | " + " | ".join(row) + " |")

    CAMPAIGN_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--domains", nargs="+")
    args = ap.parse_args()
    models = MODELS
    if args.models:
        want = set(args.models)
        models = [m for m in MODELS if m[0] in want]
    domains = tuple(args.domains) if args.domains else DOMAINS

    analysis = build_reports(models, domains)
    n = sum(len(v) for v in analysis.values())
    print(f"analyzed {n} (model, domain) pairs across {len(analysis)} models")
    print(f"wrote {CAMPAIGN_JSON} and {CAMPAIGN_MD}")


if __name__ == "__main__":
    main()
