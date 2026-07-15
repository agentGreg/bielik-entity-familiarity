"""Entity-disjoint cross-language transfer.

The language-transfer result trains
and tests the probe on the SAME entities (only the question stem changes), so it
shows ranking stability under a stem swap, not transfer to unseen entities. This
script re-runs RQ1 with entity-DISJOINT folds: train on Polish prompts for one
entity set, test on English prompts for a held-out entity set (and reverse). PL
and EN npz rows are identically ordered (verified), so row i is the same entity
across languages; splitting row indices gives entity-disjoint aligned sets.

For each direction we report within-language AUROC on the SAME held-out split and
the cross-language AUROC, so retention = cross/within is a like-for-like ratio.

Outputs results/paper2_robustness/entity_disjoint_transfer.{json,md}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper2_robustness"
OUT.mkdir(parents=True, exist_ok=True)

SLUGS = ["Bielik-1.5B-v3.0-Instruct", "Bielik-4.5B-v3.0-Instruct",
         "Bielik-Minitron-7B-v3.0-Instruct", "Bielik-11B-v3.0-Instruct",
         "gemma-4-E4B-it", "gemma-4-12b-it"]
SHORT = {"Bielik-1.5B-v3.0-Instruct": "Bielik-1.5B",
         "Bielik-4.5B-v3.0-Instruct": "Bielik-4.5B",
         "Bielik-Minitron-7B-v3.0-Instruct": "Bielik-7B",
         "Bielik-11B-v3.0-Instruct": "Bielik-11B",
         "gemma-4-E4B-it": "Gemma-E4B", "gemma-4-12b-it": "Gemma-12B"}
DOMAINS = ["athletes", "cities", "writers", "musicians"]
CONTRASTS = [("K_vs_F", "FABRICATED", "KNOWN"),
             ("K_vs_UR", "UNKNOWN_REAL", "KNOWN")]
SEED = 0


def rdir(slug, domain, lang):
    base = ROOT / "results" / slug
    if lang == "en":
        return base / "domains" / f"{domain}_en"
    return base if domain == "athletes" else base / "domains" / domain


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


def load_model(slug):
    """Return {lang: [layer -> (n_rows, dim)]}, conditions, domain_labels.
    Rows pooled across domains; PL and EN share row order (asserted)."""
    per_lang = {"pl": None, "en": None}
    conds = None; doms = None
    for lang in ("pl", "en"):
        layer_stacks = None; c_all = []; d_all = []
        for dom in DOMAINS:
            z = np.load(rdir(slug, dom, lang) / "hidden_states.npz",
                        allow_pickle=False)
            keys = sorted((k for k in z.files if k.startswith("prompt_layer_")),
                          key=lambda k: int(k.split("_")[-1]))
            hs = [np.asarray(z[k], np.float32) for k in keys]
            if layer_stacks is None:
                layer_stacks = [[] for _ in hs]
            for li, h in enumerate(hs):
                layer_stacks[li].append(h)
            c_all.append(z["conditions"])
            d_all.append(np.array([dom] * len(z["conditions"])))
        per_lang[lang] = [np.vstack(s) for s in layer_stacks]
        c = np.concatenate(c_all); d = np.concatenate(d_all)
        if conds is None:
            conds, doms = c, d
        else:
            assert np.array_equal(conds, c), f"{slug}: PL/EN misaligned"
    return per_lang, conds, doms


def transfer_direction(train_lang, test_lang, hid, layers_n, y, strat):
    """5-fold entity-disjoint. Layer picked by within-train-language OOF AUROC.
    Returns within(cross-lang-same-split) and cross AUROC as OOF over folds."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), strat))
    # pick best layer by OOF within-language AUROC on the training language
    best_layer, best_a = 0, -1.0
    oof_within_by_layer = {}
    for L in range(layers_n):
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000))
            clf.fit(hid[train_lang][L][tr], y[tr])
            oof[te] = clf.predict_proba(hid[train_lang][L][te])[:, 1]
        a = auroc(y, oof)
        oof_within_by_layer[L] = a
        if a > best_a:
            best_a, best_layer = a, L
    # at best layer: OOF within (test_lang == train_lang split) and cross
    L = best_layer
    oof_within = np.full(len(y), np.nan)
    oof_cross = np.full(len(y), np.nan)
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(hid[train_lang][L][tr], y[tr])
        oof_within[te] = clf.predict_proba(hid[train_lang][L][te])[:, 1]
        oof_cross[te] = clf.predict_proba(hid[test_lang][L][te])[:, 1]
    return {"best_layer": int(L),
            "within_auroc": round(float(auroc(y, oof_within)), 3),
            "cross_auroc": round(float(auroc(y, oof_cross)), 3)}


def main():
    results = {}
    md = ["# Entity-disjoint cross-language transfer\n",
          "Train on one language for one entity set, test on the other language "
          "for a HELD-OUT entity set (5-fold, entity-disjoint). Within-language is "
          "on the same held-out split, so retention = cross/within is like-for-like. "
          "Contrast with the paper's same-entity 96-101%.\n",
          "| model | contrast | dir | within | cross | retention |",
          "|---|---|---|---|---|---|"]
    for slug in SLUGS:
        hid, conds, doms = load_model(slug)
        layers_n = len(hid["pl"])
        results[SHORT[slug]] = {}
        for cname, neg, pos in CONTRASTS:
            mask = np.isin(conds, [pos, neg])
            y = (conds[mask] == pos).astype(int)
            strat = np.array([f"{c}|{d}" for c, d in
                              zip(conds[mask], doms[mask])])
            hid_c = {lang: [h[mask] for h in hid[lang]] for lang in ("pl", "en")}
            entry = {}
            for tl, sl, tag in [("pl", "en", "PL->EN"), ("en", "pl", "EN->PL")]:
                r = transfer_direction(tl, sl, hid_c, layers_n, y, strat)
                ret = round(r["cross_auroc"] / r["within_auroc"], 3) \
                    if r["within_auroc"] else None
                entry[tag] = {**r, "retention": ret}
                md.append(f"| {SHORT[slug]} | {cname} | {tag} | "
                          f"{r['within_auroc']} | {r['cross_auroc']} | {ret} |")
            results[SHORT[slug]][cname] = entry
    (OUT / "entity_disjoint_transfer.json").write_text(json.dumps(results, indent=2))
    # summary line
    rets = [v2["retention"] for m in results.values() for c in m.values()
            for v2 in c.values() if v2["retention"]]
    md.append(f"\nRetention range across all model/contrast/direction cells: "
              f"{min(rets):.2f}-{max(rets):.2f} (median {np.median(rets):.2f}).")
    (OUT / "entity_disjoint_transfer.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
