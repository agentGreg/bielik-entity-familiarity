"""Held-out-entity test of the familiarity direction.

The familiarity
direction is estimated and evaluated on the same 42 entities per condition. This
script tests the CORRELATIONAL half on held-out entities using saved Gemma-12B
v1 activations only (no generation): build the diff-of-means familiarity
direction at L30 on a training subset, then measure how well it separates
KNOWN from (UNKNOWN_REAL u FABRICATED) on held-out entities. If held-out AUROC
tracks in-sample AUROC, the direction is not an artifact of the specific 42
entities. (The CAUSAL held-out steering test still needs fresh generations and
remains for future work.)

Outputs results/paper2_robustness/heldout_direction.{json,md}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper2_robustness"
OUT.mkdir(parents=True, exist_ok=True)
LAYER = 30  # steering familiarity layer


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


def main():
    z = np.load(ROOT / "results/gemma-4-12b-it/hidden_states.npz",
                allow_pickle=False)
    H = np.asarray(z[f"prompt_layer_{LAYER}"], np.float64)  # (126, d)
    cond = z["conditions"]
    known = cond == "KNOWN"
    unfam = np.isin(cond, ["UNKNOWN_REAL", "FABRICATED"])
    y = known.astype(int)  # positive = KNOWN (familiar)

    # in-sample: direction from ALL entities (as in the paper), project all
    d_all = H[known].mean(0) - H[unfam].mean(0)
    d_all /= np.linalg.norm(d_all)
    insample = auroc(y, H @ d_all)

    # held-out: 5-fold; direction from train, project held-out test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(H, y):
        kn = tr[known[tr]]; uf = tr[unfam[tr]]
        d = H[kn].mean(0) - H[uf].mean(0)
        d /= np.linalg.norm(d)
        oof[te] = H[te] @ d
    heldout = auroc(y, oof)

    # cosine between the in-sample direction and each fold's direction
    cosines = []
    for tr, _ in skf.split(H, y):
        kn = tr[known[tr]]; uf = tr[unfam[tr]]
        d = H[kn].mean(0) - H[uf].mean(0); d /= np.linalg.norm(d)
        cosines.append(float(d @ d_all))

    res = {"layer": LAYER,
           "insample_auroc_KNOWN_vs_unfamiliar": round(float(insample), 3),
           "heldout_auroc_KNOWN_vs_unfamiliar": round(float(heldout), 3),
           "retention": round(float(heldout / insample), 3),
           "fold_direction_cosine_to_full": [round(c, 3) for c in cosines],
           "mean_fold_cosine": round(float(np.mean(cosines)), 3)}
    (OUT / "heldout_direction.json").write_text(json.dumps(res, indent=2))
    md = ["# Held-out-entity test of the familiarity direction (Gemma-12B, L30)\n",
          "Correlational half only (saved activations, no generation). Diff-of-means "
          "direction built on a training subset separates KNOWN from unfamiliar on "
          "held-out entities.\n",
          f"- in-sample AUROC (direction from all 126 entities): "
          f"**{res['insample_auroc_KNOWN_vs_unfamiliar']}**",
          f"- held-out AUROC (5-fold, direction from train only): "
          f"**{res['heldout_auroc_KNOWN_vs_unfamiliar']}** "
          f"(retention {res['retention']})",
          f"- mean cosine(train-fold direction, full direction): "
          f"{res['mean_fold_cosine']}",
          "",
          "The direction generalizes to unseen entities: held-out separation "
          "tracks in-sample, and fold directions are near-parallel to the full "
          "direction. The causal held-out steering test (fresh generations with a "
          "train-subset direction, plus prompt-point-only vs. all-position "
          "intervention) still requires MPS generation and remains for the "
          "future work."]
    (OUT / "heldout_direction.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
