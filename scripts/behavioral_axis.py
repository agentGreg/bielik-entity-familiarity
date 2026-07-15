"""E1 — behavioral axis: correct-vs-hallucination AUROC within KNOWN condition.

Primary target: Bielik-11B-v3.0-Instruct, where the soft judge unlocks enough
positives. Definition (primary): hallucination=1 if NOT correct_ge4, else 0
(i.e. fewer than 4 of 5 sampled answers pass the soft rubric). correct_5of5
and correct_ge3 are computed as sensitivity thresholds.

For 11B we compute, within KNOWN rows only:
  - AUROC per MLP layer of ipr / entropy at point='prompt' (results/<slug>/signals.parquet),
    via auroc_per_layer; report max over layers of max(auc, 1-auc).
  - first_token_entropy AUROC (constant across layers, computed once) as a
    zero-cost baseline.
  - Linear-probe AUROC per residual-stream layer (results/<slug>/hidden_states.npz,
    keys prompt_layer_<k>) via probe_auroc_per_layer (5-fold CV), plus a
    shuffled-label floor (same probe, permuted labels, seed 0).

We also sweep 4.5B and 1.5B at whatever threshold yields >=8 positives (if
any), flagging underpowered results (n_pos < 8).

Row alignment: hidden_states.npz's 'conditions' array and labeled.parquet
rows share order; we assert this explicitly. Entities are matched between
labeled.parquet and labeled_soft.parquet on the 'entity' column (soft judge
only has KNOWN rows, so this also gives us the KNOWN row mask/order).

Output: results/behavioral_axis_11B.json — despite the filename (kept as
specified in the E1 task), this file contains per-model results under
per-model keys, with '11B' as the primary/complete analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bielik_hallu.analysis.auroc import auroc, auroc_per_layer
from bielik_hallu.analysis.probe import probe_auroc_per_layer

ROOT = Path(__file__).resolve().parents[1]
MODEL_SLUGS = [
    "Bielik-1.5B-v3.0-Instruct",
    "Bielik-4.5B-v3.0-Instruct",
    "Bielik-11B-v3.0-Instruct",
]
THRESHOLDS = ("correct_5of5", "correct_ge4", "correct_ge3")
MIN_POSITIVES = 8
SHUFFLE_SEED = 0


def load_known_with_soft_labels(slug: str) -> pd.DataFrame:
    """Return labeled.parquet KNOWN rows merged with labeled_soft.parquet on
    entity, preserving labeled.parquet's KNOWN row order (which matches the
    npz row order after filtering by condition == 'KNOWN')."""
    labeled = pd.read_parquet(ROOT / "data" / slug / "labeled.parquet")
    soft = pd.read_parquet(ROOT / "data" / slug / "labeled_soft.parquet")

    known = labeled[labeled["condition"] == "KNOWN"].reset_index(drop=True)
    known = known.reset_index().rename(columns={"index": "_row_in_labeled"})

    merged = known.merge(
        soft[["entity", "n_correct_soft", "soft_verdicts",
              "correct_5of5", "correct_ge4", "correct_ge3"]],
        on="entity", how="left", validate="one_to_one",
    )
    missing = merged["n_correct_soft"].isna().sum()
    if missing:
        raise ValueError(f"[{slug}] {missing} KNOWN entities missing from labeled_soft.parquet")
    return merged


def dispersion_auroc_for_slug(slug: str, known_idx_in_labeled: np.ndarray, labels: np.ndarray) -> dict:
    """AUROC per MLP layer of ipr/entropy at point='prompt', restricted to the
    given KNOWN rows (identified by entity), plus first_token_entropy baseline."""
    signals = pd.read_parquet(ROOT / "results" / slug / "signals.parquet")
    prompt_sig = signals[signals["point"] == "prompt"]

    labeled = pd.read_parquet(ROOT / "data" / slug / "labeled.parquet")
    known_entities = labeled.iloc[known_idx_in_labeled]["entity"].to_numpy()
    entity_to_label = dict(zip(known_entities, labels))

    known_sig = prompt_sig[
        (prompt_sig["condition"] == "KNOWN") & (prompt_sig["entity"].isin(known_entities))
    ].copy()
    known_sig["_label"] = known_sig["entity"].map(entity_to_label)

    result = {}
    for metric in ("ipr", "entropy"):
        # Build metric-by-layer and matching label array (same entity order per layer).
        by_layer = {}
        for layer, g in known_sig.groupby("layer"):
            g_sorted = g.sort_values("entity")
            by_layer[layer] = g_sorted[metric].to_numpy()
        # labels sorted by entity to match
        first_layer = next(iter(known_sig["layer"].unique()))
        g0 = known_sig[known_sig["layer"] == first_layer].sort_values("entity")
        label_arr = g0["_label"].to_numpy()

        aucs = auroc_per_layer(by_layer, label_arr)
        valid = {k: v for k, v in aucs.items() if not np.isnan(v)}
        best_layer = max(valid, key=valid.get) if valid else None
        result[metric] = {
            "best_auroc": valid[best_layer] if best_layer is not None else float("nan"),
            "best_layer": int(best_layer) if best_layer is not None else None,
        }

    # first_token_entropy: constant across layers -> compute once on any single layer's rows.
    g0 = known_sig[known_sig["layer"] == next(iter(known_sig["layer"].unique()))].sort_values("entity")
    fte_scores = g0["first_token_entropy"].to_numpy()
    fte_labels = g0["_label"].to_numpy()
    result["first_token_entropy_baseline"] = auroc(fte_scores, fte_labels)
    return result


def probe_auroc_for_slug(slug: str, known_idx_in_labeled: np.ndarray, labels: np.ndarray) -> dict:
    """Linear-probe AUROC per residual-stream layer restricted to KNOWN rows
    identified by known_idx_in_labeled (positions within the full 123-row
    labeled.parquet / npz array), plus a shuffled-label floor."""
    npz_path = ROOT / "results" / slug / "hidden_states.npz"
    data = np.load(npz_path, allow_pickle=False)

    labeled = pd.read_parquet(ROOT / "data" / slug / "labeled.parquet")
    npz_conditions = data["conditions"]
    labeled_conditions = labeled["condition"].to_numpy()
    if not np.array_equal(npz_conditions, labeled_conditions):
        raise ValueError(f"[{slug}] npz 'conditions' order does not match labeled.parquet order")

    layer_keys = sorted(
        (k for k in data.files if k.startswith("prompt_layer_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    hidden_by_layer = {
        int(k.split("_")[-1]): data[k][known_idx_in_labeled] for k in layer_keys
    }

    probe_aucs = probe_auroc_per_layer(hidden_by_layer, labels, seed=SHUFFLE_SEED)
    valid = {k: v for k, v in probe_aucs.items() if not np.isnan(v)}
    best_layer = max(valid, key=valid.get) if valid else None

    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)
    shuffled_aucs = probe_auroc_per_layer(hidden_by_layer, shuffled_labels, seed=SHUFFLE_SEED)
    shuffled_valid = {k: v for k, v in shuffled_aucs.items() if not np.isnan(v)}
    shuffled_best = max(shuffled_valid.values()) if shuffled_valid else float("nan")

    return {
        "best_auroc": valid[best_layer] if best_layer is not None else float("nan"),
        "best_layer": int(best_layer) if best_layer is not None else None,
        "shuffled_floor_best": shuffled_best,
    }


def run_threshold(slug: str, merged: pd.DataFrame, threshold_col: str) -> dict:
    correct = merged[threshold_col].to_numpy().astype(bool)
    labels = (~correct).astype(int)  # hallucination = 1 if NOT correct
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    known_idx_in_labeled = merged["_row_in_labeled"].to_numpy()

    entry = {
        "threshold": threshold_col,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "underpowered": n_pos < MIN_POSITIVES or n_neg < MIN_POSITIVES,
    }

    if len(np.unique(labels)) < 2:
        entry["note"] = "degenerate label (single class) — AUROC undefined"
        return entry

    try:
        dispersion = dispersion_auroc_for_slug(slug, known_idx_in_labeled, labels)
        entry["dispersion"] = dispersion
    except Exception as exc:  # noqa: BLE001
        entry["dispersion_error"] = str(exc)

    try:
        probe = probe_auroc_for_slug(slug, known_idx_in_labeled, labels)
        entry["probe"] = probe
    except Exception as exc:  # noqa: BLE001
        entry["probe_error"] = str(exc)

    return entry


def main() -> None:
    output = {}
    for slug in MODEL_SLUGS:
        soft_path = ROOT / "data" / slug / "labeled_soft.parquet"
        if not soft_path.exists():
            output[slug] = {"error": f"missing {soft_path}; run scripts/soft_judge.py first"}
            print(f"[{slug}] SKIP — {soft_path} not found")
            continue

        merged = load_known_with_soft_labels(slug)
        model_out = {"n_known": len(merged), "thresholds": {}}
        for threshold_col in THRESHOLDS:
            print(f"[{slug}] threshold={threshold_col}")
            entry = run_threshold(slug, merged, threshold_col)
            model_out["thresholds"][threshold_col] = entry
            print(f"  n_pos={entry['n_pos']} n_neg={entry['n_neg']} underpowered={entry['underpowered']}")
            if "dispersion" in entry:
                print(f"  dispersion: {entry['dispersion']}")
            if "probe" in entry:
                print(f"  probe: {entry['probe']}")
        output[slug] = model_out

    out_path = ROOT / "results" / "behavioral_axis_11B.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
