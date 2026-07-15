"""E6/E7 — cross-domain generalization of the activation-dispersion signal.

Four domains (athletes, cities, writers, musicians) x four models
(Bielik 1.5B / 4.5B / Minitron-7B / 11B v3.0-Instruct). Every domain has
condition-only labels for 42 entities per condition (KNOWN / UNKNOWN_REAL /
FABRICATED), so all three contrasts are available.

Sections:
  (a) Per-domain, per-model separability:
        - best dispersion AUROC (max over {ipr, entropy} and layers) + bootstrap
          CI at the fixed best (metric, layer)
        - probe AUROC per layer (5-fold CV), best layer reported
        - first-token-entropy AUROC
        for K-vs-F (headline), K-vs-UR (anti-confound), UR-vs-F (lexical upper bound)
  (b) Ferrando-style transfer matrix (per model):
        - probe transfer: train on domain A's K+F at A's CV-best layer, evaluate
          zero-shot on domain B's K+F at the SAME layer (diagonal = within-domain CV)
        - dispersion transfer: pick best (metric, layer) on A, evaluate that exact
          (metric, layer) on B
      Mean off-diagonal per model; cities-template effect flagged.
  (c) Layer-band stability: relative-depth band (AUROC >= 0.9 x max) per
      domain x model x metric-family, and cross-domain overlap of the band.
  (d) Per-head analysis (E7): rank heads by K-vs-F AUROC of attention entropy at
      the prompt point; top-10 per model per domain; cross-domain consistency
      (Spearman of per-head AUROC vectors, top-20 overlap); relative depth of top
      heads.

Athletes signals/hidden/attn live at results/<slug>/; the three new domains at
results/<slug>/domains/<domain>/. The KNOWN/UNKNOWN_REAL/FABRICATED conditions and
the npz row order are shared with data/<slug>/{labeled or domains/<d>/labeled}.parquet.

Outputs (the only files this script writes):
  results/domain_generalization.json
  results/domain_generalization.md

Usage:
  .venv/bin/python scripts/domain_generalization.py

Env knobs:
  DG_N_BOOT   bootstrap resamples for the best-cell CI (default 10000)
  DG_N_JOBS   joblib workers (default -1)

All randomness: numpy default_rng(0); sklearn CV random_state=0.
Separability convention throughout: max(auc, 1-auc).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bielik_hallu.analysis.probe import probe_auroc_per_layer  # noqa: E402

SLUGS = [
    "Bielik-1.5B-v3.0-Instruct",
    "Bielik-4.5B-v3.0-Instruct",
    "Bielik-Minitron-7B-v3.0-Instruct",
    "Bielik-11B-v3.0-Instruct",
]
SHORT = {
    "Bielik-1.5B-v3.0-Instruct": "1.5B",
    "Bielik-4.5B-v3.0-Instruct": "4.5B",
    "Bielik-Minitron-7B-v3.0-Instruct": "7B (Minitron)",
    "Bielik-11B-v3.0-Instruct": "11B",
}
DOMAINS = ["athletes", "cities", "writers", "musicians"]
# cities uses a "Czym jest X?" (what is) template; the other three use "Kim jest X?"
# (who is). Transfer to/from cities therefore also carries a prompt-template shift.
PEOPLE_DOMAINS = {"athletes", "writers", "musicians"}
TEMPLATE_ODD = "cities"

CONTRASTS = [
    ("K_vs_F", "FABRICATED", "KNOWN"),
    ("K_vs_UR", "UNKNOWN_REAL", "KNOWN"),
    ("UR_vs_F", "FABRICATED", "UNKNOWN_REAL"),
]

SEED = 0
N_BOOT = int(os.environ.get("DG_N_BOOT", 10_000))
N_JOBS = int(os.environ.get("DG_N_JOBS", -1))
BAND_FRAC = 0.9  # AUROC >= BAND_FRAC * max defines the "knowledge band"

JSON_PATH = ROOT / "results" / "domain_generalization.json"
MD_PATH = ROOT / "results" / "domain_generalization.md"


# --------------------------------------------------------------------------
# Path resolution + loading (athletes at root, others under domains/<d>/)
# --------------------------------------------------------------------------

def _results_dir(slug: str, domain: str) -> Path:
    return ROOT / "results" / slug if domain == "athletes" else ROOT / "results" / slug / "domains" / domain


def load_prompt_signals(slug: str, domain: str):
    """(layers, entities, conditions, mats, fte) at point='prompt'."""
    sig = pd.read_parquet(_results_dir(slug, domain) / "signals.parquet")
    p = sig[sig["point"] == "prompt"]
    entities = np.array(sorted(p["entity"].unique()))
    layers = sorted(p["layer"].unique())
    mats = {}
    for metric in ("ipr", "entropy"):
        piv = p.pivot(index="layer", columns="entity", values=metric)
        mats[metric] = piv.loc[layers, entities].to_numpy(dtype=np.float64)
    per_ent = p.drop_duplicates("entity").set_index("entity")
    conditions = per_ent["condition"].loc[entities].to_numpy()
    fte = per_ent["first_token_entropy"].loc[entities].to_numpy(dtype=np.float64)
    return layers, entities, conditions, mats, fte


def load_prompt_hidden(slug: str, domain: str):
    """(layer_ids, hidden_list, conditions): residual-stream prompt-point states."""
    z = np.load(_results_dir(slug, domain) / "hidden_states.npz", allow_pickle=False)
    keys = sorted(
        (k for k in z.files if k.startswith("prompt_layer_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    layer_ids = [int(k.split("_")[-1]) for k in keys]
    hidden = [np.asarray(z[k], dtype=np.float32) for k in keys]
    return layer_ids, hidden, z["conditions"]


def load_attn_per_head(slug: str, domain: str):
    return pd.read_parquet(_results_dir(slug, domain) / "attn_per_head.parquet")


def contrast_mask(conditions: np.ndarray, pos: str, neg: str):
    mask = np.isin(conditions, [pos, neg])
    labels = (conditions[mask] == pos).astype(int)
    return mask, labels


# --------------------------------------------------------------------------
# AUROC machinery
# --------------------------------------------------------------------------

def auc_matrix(M: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Plain AUROC for every row of M (n_layers, n) against binary labels."""
    R = rankdata(M, axis=1)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    s = R[:, labels == 1].sum(axis=1)
    return (s - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def sep(a):
    return np.maximum(a, 1.0 - a)


def bootstrap_auroc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int, rng) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    obs = roc_auc_score(labels, scores)
    if obs < 0.5:
        scores = -scores
        obs = 1.0 - obs
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, pos.size, replace=True),
            rng.choice(neg, neg.size, replace=True),
        ])
        boots[i] = roc_auc_score(labels[idx], scores[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": float(obs), "ci95": [float(lo), float(hi)], "n_boot": n_boot}


def _probe_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def probe_sweep(hidden_list, labels: np.ndarray, seed: int = SEED):
    """Per-layer CV probe separability. Returns (aucs array, best_idx)."""
    hb = {i: X for i, X in enumerate(hidden_list)}
    res = probe_auroc_per_layer(hb, labels, seed=seed)
    aucs = np.array([res[i] for i in range(len(hidden_list))])
    best = int(np.nanargmax(aucs))
    return aucs, best


# --------------------------------------------------------------------------
# (a) per-domain per-model separability
# --------------------------------------------------------------------------

def section_a(rng) -> dict:
    out = {}
    for slug in SLUGS:
        out[slug] = {}
        for domain in DOMAINS:
            layers, entities, conditions, mats, fte = load_prompt_signals(slug, domain)
            layer_ids, hidden, npz_cond = load_prompt_hidden(slug, domain)
            dom_cell = {}
            for cname, pos, neg in CONTRASTS:
                mask, labels = contrast_mask(conditions, pos, neg)
                cell = {}
                # dispersion: best over {ipr, entropy} and layers, CI at fixed best cell
                best_metric, best_idx, best_auc = None, None, -1.0
                per_metric = {}
                for metric in ("ipr", "entropy"):
                    aucs = sep(auc_matrix(mats[metric][:, mask], labels))
                    bi = int(np.argmax(aucs))
                    per_metric[metric] = {"auroc": float(aucs[bi]), "best_layer": int(layers[bi])}
                    if aucs[bi] > best_auc:
                        best_metric, best_idx, best_auc = metric, bi, float(aucs[bi])
                ci = bootstrap_auroc_ci(mats[best_metric][best_idx, mask], labels, N_BOOT, rng)
                ci["best_layer"] = int(layers[best_idx])
                ci["best_metric"] = best_metric
                cell["dispersion_best"] = ci
                cell["dispersion_per_metric"] = per_metric

                # probe: per-layer CV, best layer
                npz_mask, npz_labels = contrast_mask(npz_cond, pos, neg)
                hidden_sub = [X[npz_mask] for X in hidden]
                probe_aucs, probe_best = probe_sweep(hidden_sub, npz_labels)
                cell["probe"] = {
                    "auroc": float(probe_aucs[probe_best]),
                    "best_layer": int(layer_ids[probe_best]),
                }
                # first-token entropy
                cell["first_token_entropy"] = {
                    "auroc": float(sep(auc_matrix(fte[mask][None, :], labels))[0]),
                }
                dom_cell[cname] = cell
            out[slug][domain] = dom_cell
    return out


# --------------------------------------------------------------------------
# (b) transfer matrices (per model)
# --------------------------------------------------------------------------

def _kf_pack(slug: str, domain: str):
    """K-vs-F hidden states (aligned across layers) + dispersion mats + labels."""
    layers, entities, conditions, mats, fte = load_prompt_signals(slug, domain)
    layer_ids, hidden, npz_cond = load_prompt_hidden(slug, domain)
    mask, labels = contrast_mask(conditions, "FABRICATED", "KNOWN")
    npz_mask, npz_labels = contrast_mask(npz_cond, "FABRICATED", "KNOWN")
    disp = {m: mats[m][:, mask] for m in ("ipr", "entropy")}
    hidden_kf = [X[npz_mask] for X in hidden]
    return {
        "layers": layers, "layer_ids": layer_ids,
        "disp": disp, "labels": labels,
        "hidden": hidden_kf, "npz_labels": npz_labels,
    }


def section_b(packs: dict) -> dict:
    """Per model: 4x4 probe transfer matrix + dispersion transfer matrix.

    packs[slug][domain] = _kf_pack(...). Diagonal:
      probe  -> within-domain CV separability (max over layers)
      disp   -> within-domain best (metric, layer) separability
    Off-diagonal A->B:
      probe  -> fit probe on ALL of A at A's CV-best layer index, score B at the
                same layer index (residual dims match within a model)
      disp   -> take A's best (metric, layer), read that exact row on B, sep-AUROC
    """
    out = {}
    for slug in SLUGS:
        dp = packs[slug]
        # precompute per-domain CV-best layer index + best dispersion (metric, layer idx)
        probe_best_idx = {}
        disp_best = {}  # domain -> (metric, layer_idx)
        for d in DOMAINS:
            aucs, best = probe_sweep(dp[d]["hidden"], dp[d]["npz_labels"])
            probe_best_idx[d] = best
            bm, bl, ba = None, None, -1.0
            for m in ("ipr", "entropy"):
                ma = sep(auc_matrix(dp[d]["disp"][m], dp[d]["labels"]))
                bi = int(np.argmax(ma))
                if ma[bi] > ba:
                    bm, bl, ba = m, bi, float(ma[bi])
            disp_best[d] = (bm, bl)

        probe_mat = {}
        disp_mat = {}
        for a in DOMAINS:
            probe_mat[a] = {}
            disp_mat[a] = {}
            a_layer = probe_best_idx[a]
            # fit probe on ALL of A at A's best layer
            Xa = dp[a]["hidden"][a_layer]
            ya = dp[a]["npz_labels"]
            clf = _probe_pipeline().fit(Xa, ya)
            am, al = disp_best[a]
            for b in DOMAINS:
                if a == b:
                    # diagonal: within-domain CV / within-domain best dispersion
                    aucs, _ = probe_sweep(dp[b]["hidden"], dp[b]["npz_labels"])
                    probe_mat[a][b] = float(aucs[a_layer])
                    disp_mat[a][b] = float(sep(auc_matrix(dp[b]["disp"][am][al:al + 1], dp[b]["labels"]))[0])
                else:
                    Xb = dp[b]["hidden"][a_layer]
                    yb = dp[b]["npz_labels"]
                    proba = clf.predict_proba(Xb)[:, 1]
                    ab = roc_auc_score(yb, proba)
                    probe_mat[a][b] = float(max(ab, 1.0 - ab))
                    disp_mat[a][b] = float(sep(auc_matrix(dp[b]["disp"][am][al:al + 1], dp[b]["labels"]))[0])

        def mean_offdiag(mat):
            vals = [mat[a][b] for a in DOMAINS for b in DOMAINS if a != b]
            return float(np.mean(vals))

        def worst_cell(mat):
            cells = [(mat[a][b], a, b) for a in DOMAINS for b in DOMAINS if a != b]
            v, a, b = min(cells)
            return {"auroc": v, "train": a, "eval": b}

        def cities_effect(mat):
            """Compare off-diagonal cells that involve cities vs those among people domains."""
            involving = [mat[a][b] for a in DOMAINS for b in DOMAINS
                         if a != b and (a == TEMPLATE_ODD or b == TEMPLATE_ODD)]
            people = [mat[a][b] for a in PEOPLE_DOMAINS for b in PEOPLE_DOMAINS if a != b]
            return {
                "mean_involving_cities": float(np.mean(involving)),
                "mean_people_to_people": float(np.mean(people)),
                "delta": float(np.mean(people) - np.mean(involving)),
            }

        out[slug] = {
            "probe_transfer": probe_mat,
            "dispersion_transfer": disp_mat,
            "probe_best_layer_idx": {d: int(probe_best_idx[d]) for d in DOMAINS},
            "probe_best_layer_id": {d: int(dp[d]["layer_ids"][probe_best_idx[d]]) for d in DOMAINS},
            "dispersion_best_selection": {
                d: {"metric": disp_best[d][0], "layer_id": int(dp[d]["layers"][disp_best[d][1]])}
                for d in DOMAINS
            },
            "probe_mean_offdiag": mean_offdiag(probe_mat),
            "dispersion_mean_offdiag": mean_offdiag(disp_mat),
            "probe_worst_cell": worst_cell(probe_mat),
            "dispersion_worst_cell": worst_cell(disp_mat),
            "probe_cities_template_effect": cities_effect(probe_mat),
            "dispersion_cities_template_effect": cities_effect(disp_mat),
        }
    return out


# --------------------------------------------------------------------------
# (c) layer-band stability (K-vs-F)
# --------------------------------------------------------------------------

def section_c() -> dict:
    """For each model x metric-family, the relative-depth band (AUROC >= 0.9*max)
    per domain, and the cross-domain intersection of those bands (on the shared
    relative-depth axis)."""
    out = {}
    for slug in SLUGS:
        out[slug] = {}
        for family in ("dispersion", "probe"):
            per_domain_bands = {}  # domain -> (lo_rel, hi_rel, best_rel)
            for domain in DOMAINS:
                layers, entities, conditions, mats, fte = load_prompt_signals(slug, domain)
                layer_ids, hidden, npz_cond = load_prompt_hidden(slug, domain)
                if family == "dispersion":
                    mask, labels = contrast_mask(conditions, "FABRICATED", "KNOWN")
                    # per-layer best over {ipr, entropy}
                    curves = np.vstack([sep(auc_matrix(mats[m][:, mask], labels)) for m in ("ipr", "entropy")])
                    curve = curves.max(axis=0)
                    lyr = np.asarray(layers, dtype=np.float64)
                else:
                    npz_mask, npz_labels = contrast_mask(npz_cond, "FABRICATED", "KNOWN")
                    hidden_sub = [X[npz_mask] for X in hidden]
                    aucs, _ = probe_sweep(hidden_sub, npz_labels)
                    curve = aucs
                    lyr = np.asarray(layer_ids, dtype=np.float64)
                mx = float(np.nanmax(curve))
                thr = BAND_FRAC * mx
                in_band = curve >= thr
                rel = (lyr - lyr.min()) / (lyr.max() - lyr.min())
                band_rel = rel[in_band]
                best_rel = float(rel[int(np.nanargmax(curve))])
                per_domain_bands[domain] = {
                    "max_auroc": mx,
                    "threshold": float(thr),
                    "band_rel_lo": float(band_rel.min()),
                    "band_rel_hi": float(band_rel.max()),
                    "band_frac_layers": float(in_band.mean()),
                    "best_rel_depth": best_rel,
                }
            # cross-domain overlap: intersection of [lo,hi] relative intervals
            los = [b["band_rel_lo"] for b in per_domain_bands.values()]
            his = [b["band_rel_hi"] for b in per_domain_bands.values()]
            inter_lo, inter_hi = max(los), min(his)
            union_lo, union_hi = min(los), max(his)
            overlap = max(0.0, inter_hi - inter_lo)
            union = union_hi - union_lo
            out[slug][family] = {
                "per_domain": per_domain_bands,
                "intersection_rel": [float(inter_lo), float(inter_hi)] if inter_hi >= inter_lo else None,
                "overlap_width": float(overlap),
                "union_width": float(union),
                "overlap_over_union": float(overlap / union) if union > 0 else 0.0,
                "best_rel_depth_spread": float(
                    max(b["best_rel_depth"] for b in per_domain_bands.values())
                    - min(b["best_rel_depth"] for b in per_domain_bands.values())
                ),
            }
    return out


# --------------------------------------------------------------------------
# (d) per-head analysis (E7)
# --------------------------------------------------------------------------

def _head_auroc_table(slug: str, domain: str) -> pd.DataFrame:
    """Return DataFrame indexed by (layer, head) with K-vs-F sep-AUROC of the
    prompt-point attention entropy, plus n_layers for relative depth."""
    a = load_attn_per_head(slug, domain)
    p = a[a["point"] == "prompt"]
    # labels per entity via condition
    ent_cond = p.drop_duplicates("entity").set_index("entity")["condition"]
    mask_ents = ent_cond[ent_cond.isin(["KNOWN", "FABRICATED"])].index
    labels_map = (ent_cond.loc[mask_ents] == "FABRICATED").astype(int)
    sub = p[p["entity"].isin(mask_ents)].copy()
    n_layers = int(sub["layer"].nunique())
    max_layer = int(sub["layer"].max())
    rows = []
    for (layer, head), g in sub.groupby(["layer", "head"]):
        g = g.sort_values("entity")
        y = labels_map.loc[g["entity"].to_numpy()].to_numpy()
        scores = g["attn_entropy"].to_numpy(dtype=np.float64)
        au = roc_auc_score(y, scores)
        rows.append((int(layer), int(head), float(max(au, 1.0 - au))))
    df = pd.DataFrame(rows, columns=["layer", "head", "auroc"])
    df["rel_depth"] = df["layer"] / max_layer if max_layer else 0.0
    return df, n_layers, max_layer


def section_d() -> dict:
    out = {}
    for slug in SLUGS:
        per_domain_df = {}
        entry = {"top10_per_domain": {}, "n_heads_total": None}
        for domain in DOMAINS:
            df, n_layers, max_layer = _head_auroc_table(slug, domain)
            per_domain_df[domain] = df
            top10 = df.sort_values("auroc", ascending=False).head(10)
            entry["top10_per_domain"][domain] = [
                {"layer": int(r.layer), "head": int(r.head),
                 "auroc": round(float(r.auroc), 4), "rel_depth": round(float(r.rel_depth), 3)}
                for r in top10.itertuples()
            ]
        entry["n_heads_total"] = int(len(per_domain_df["athletes"]))

        # cross-domain consistency: align per-head auroc vectors on (layer, head)
        base_keys = per_domain_df["athletes"][["layer", "head"]].apply(tuple, axis=1)
        vecs = {}
        for domain in DOMAINS:
            d = per_domain_df[domain].set_index(["layer", "head"])["auroc"]
            vecs[domain] = d
        common = None
        for domain in DOMAINS:
            idx = set(vecs[domain].index)
            common = idx if common is None else (common & idx)
        common = sorted(common)
        aligned = {domain: np.array([vecs[domain][k] for k in common]) for domain in DOMAINS}

        # Spearman correlation between every domain pair
        spearman = {}
        for i, a in enumerate(DOMAINS):
            for b in DOMAINS[i + 1:]:
                rho, _ = spearmanr(aligned[a], aligned[b])
                spearman[f"{a}~{b}"] = round(float(rho), 4)
        entry["spearman_pairwise"] = spearman
        entry["spearman_mean"] = round(float(np.mean(list(spearman.values()))), 4)

        # top-20 overlap (Jaccard) between domain pairs, and heads in ALL four top-20
        top20 = {}
        for domain in DOMAINS:
            df = per_domain_df[domain].sort_values("auroc", ascending=False).head(20)
            top20[domain] = set(df[["layer", "head"]].apply(tuple, axis=1))
        overlap = {}
        for i, a in enumerate(DOMAINS):
            for b in DOMAINS[i + 1:]:
                inter = len(top20[a] & top20[b])
                union = len(top20[a] | top20[b])
                overlap[f"{a}~{b}"] = {"intersection": inter, "jaccard": round(inter / union, 3)}
        entry["top20_overlap"] = overlap
        in_all_four = top20["athletes"] & top20["cities"] & top20["writers"] & top20["musicians"]
        entry["heads_in_all_four_top20"] = sorted(
            ({"layer": int(l), "head": int(h),
              "mean_auroc": round(float(np.mean([vecs[dm][(l, h)] for dm in DOMAINS])), 4)}
             for (l, h) in in_all_four),
            key=lambda x: -x["mean_auroc"],
        )

        # mean per-head auroc across domains -> the most consistent heads
        mean_vec = np.mean([aligned[dm] for dm in DOMAINS], axis=0)
        order = np.argsort(-mean_vec)[:10]
        max_layer = per_domain_df["athletes"]["layer"].max()
        entry["top10_by_mean_auroc"] = [
            {"layer": int(common[i][0]), "head": int(common[i][1]),
             "mean_auroc": round(float(mean_vec[i]), 4),
             "rel_depth": round(float(common[i][0] / max_layer), 3),
             "per_domain": {dm: round(float(vecs[dm][common[i]]), 4) for dm in DOMAINS}}
            for i in order
        ]
        # concentration diagnostic: how much of the K-vs-F signal is in the top-k heads
        # (max single-head auroc vs the 10th, 50th percentile across all heads, per domain)
        conc = {}
        for domain in DOMAINS:
            v = np.sort(per_domain_df[domain]["auroc"].to_numpy())[::-1]
            conc[domain] = {
                "max": round(float(v[0]), 4),
                "top10_min": round(float(v[9]), 4),
                "median": round(float(np.median(v)), 4),
                "frac_heads_ge_0.8": round(float((v >= 0.8).mean()), 4),
                "frac_heads_ge_0.9": round(float((v >= 0.9).mean()), 4),
            }
        entry["concentration"] = conc
        out[slug] = entry
    return out


# --------------------------------------------------------------------------
# JSON/markdown output
# --------------------------------------------------------------------------

def _san(obj):
    if isinstance(obj, dict):
        return {str(k): _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _fmt_ci(c, digits=3):
    if not c or c.get("auroc") is None:
        return "n/a"
    s = f"{c['auroc']:.{digits}f}"
    if "best_layer" in c:
        s += f"@L{c['best_layer']}"
    if "ci95" in c:
        s += f" [{c['ci95'][0]:.{digits}f}, {c['ci95'][1]:.{digits}f}]"
    return s


def render_markdown(res: dict) -> str:
    meta = res["meta"]
    L: list[str] = []
    L.append("# E6/E7 — Cross-domain generalization of the dispersion signal")
    L.append("")
    L.append(f"Generated by `scripts/domain_generalization.py`. "
             f"Models: {', '.join(SHORT[s] for s in SLUGS)}. "
             f"Domains: {', '.join(DOMAINS)} (42 entities x 3 conditions each). "
             f"Seeds: numpy `default_rng({SEED})`, sklearn CV `random_state={SEED}`. "
             f"Bootstrap: {meta['n_boot']} resamples. "
             "Separability convention: max(AUROC, 1-AUROC). Point: prompt (last prompt token). "
             f"Layer band: AUROC >= {BAND_FRAC:g} x max on the relative-depth axis.")
    L.append("")
    L.append(f"**Template note.** `cities` uses a *Czym jest X?* (\"what is\") prompt; "
             "`athletes`/`writers`/`musicians` use *Kim jest X?* (\"who is\"). Transfer "
             "to/from cities therefore carries a prompt-template shift as well as an "
             "entity-type shift; this is quantified in the transfer section.")
    L.append("")

    # ---- (a) per-domain separability -----------------------------------
    L.append("## (a) Per-domain, per-model separability")
    L.append("")
    has_probe_ci = any(
        "ci95" in res["a_per_domain"][s][d]["K_vs_F"]["probe"]
        for s in SLUGS for d in DOMAINS
    )
    if has_probe_ci:
        L.append("Probe cells in the K-vs-F table carry 95% bootstrap CIs computed by "
                 "resampling the out-of-fold 5-fold-CV probabilities at the FIXED best "
                 "layer (the fixed-best-layer convention; 10k stratified resamples, "
                 "numpy `default_rng(0)`).")
        L.append("")
    for cname, title in (("K_vs_F", "KNOWN vs FABRICATED (headline)"),
                         ("K_vs_UR", "KNOWN vs UNKNOWN-REAL (anti-confound)"),
                         ("UR_vs_F", "UNKNOWN-REAL vs FABRICATED (lexical upper bound)")):
        L.append(f"### {title}")
        L.append("")
        L.append("| Model | Domain | Best dispersion (CI) | Probe | First-token entropy |")
        L.append("|---|---|---|---|---|")
        for slug in SLUGS:
            for domain in DOMAINS:
                c = res["a_per_domain"][slug][domain][cname]
                L.append(f"| {SHORT[slug]} | {domain} | {_fmt_ci(c['dispersion_best'])} "
                         f"({c['dispersion_best']['best_metric']}) | {_fmt_ci(c['probe'])} | "
                         f"{c['first_token_entropy']['auroc']:.3f} |")
        L.append("")

    # ---- (b) transfer matrices -----------------------------------------
    L.append("## (b) Transfer matrices (Ferrando-style, K-vs-F)")
    L.append("")
    L.append("Rows = train domain A, columns = eval domain B. Probe: fit on ALL of A at "
             "A's CV-best layer, score B at the same layer index (diagonal = within-domain "
             "5-fold CV at that layer). Dispersion: A's best (metric, layer) evaluated on B "
             "(diagonal = within-domain best).")
    L.append("")
    for slug in SLUGS:
        t = res["b_transfer"][slug]
        L.append(f"### {SHORT[slug]} — probe transfer")
        L.append("")
        L.append("| train \\ eval | " + " | ".join(DOMAINS) + " |")
        L.append("|---|" + "|".join(["---"] * len(DOMAINS)) + "|")
        for a in DOMAINS:
            cells = " | ".join(f"{t['probe_transfer'][a][b]:.3f}" for b in DOMAINS)
            L.append(f"| **{a}** | {cells} |")
        L.append("")
        L.append(f"Mean off-diagonal (probe): **{t['probe_mean_offdiag']:.3f}**; "
                 f"worst cell {t['probe_worst_cell']['auroc']:.3f} "
                 f"({t['probe_worst_cell']['train']}->{t['probe_worst_cell']['eval']}). "
                 f"Cities-template effect: people->people {t['probe_cities_template_effect']['mean_people_to_people']:.3f} "
                 f"vs cities-involving {t['probe_cities_template_effect']['mean_involving_cities']:.3f} "
                 f"(delta {t['probe_cities_template_effect']['delta']:+.3f}).")
        L.append("")
        L.append(f"### {SHORT[slug]} — dispersion transfer")
        L.append("")
        L.append("| train \\ eval | " + " | ".join(DOMAINS) + " |")
        L.append("|---|" + "|".join(["---"] * len(DOMAINS)) + "|")
        for a in DOMAINS:
            cells = " | ".join(f"{t['dispersion_transfer'][a][b]:.3f}" for b in DOMAINS)
            L.append(f"| **{a}** | {cells} |")
        L.append("")
        L.append(f"Mean off-diagonal (dispersion): **{t['dispersion_mean_offdiag']:.3f}**; "
                 f"worst cell {t['dispersion_worst_cell']['auroc']:.3f} "
                 f"({t['dispersion_worst_cell']['train']}->{t['dispersion_worst_cell']['eval']}). "
                 f"Cities-template effect: people->people "
                 f"{t['dispersion_cities_template_effect']['mean_people_to_people']:.3f} vs "
                 f"cities-involving {t['dispersion_cities_template_effect']['mean_involving_cities']:.3f} "
                 f"(delta {t['dispersion_cities_template_effect']['delta']:+.3f}).")
        L.append("")

    # ---- (c) layer band ------------------------------------------------
    L.append("## (c) Layer-band stability across domains (K-vs-F)")
    L.append("")
    L.append(f"Relative-depth band where AUROC >= {BAND_FRAC:g} x max, per domain, and the "
             "cross-domain intersection. `overlap/union` near 1 means the knowledge band sits "
             "at the same relative depth for every entity type.")
    L.append("")
    for family in ("dispersion", "probe"):
        L.append(f"### {family}")
        L.append("")
        L.append("| Model | athletes | cities | writers | musicians | Intersection | overlap/union | best-depth spread |")
        L.append("|---|---|---|---|---|---|---|---|")
        for slug in SLUGS:
            e = res["c_layer_band"][slug][family]
            pd_ = e["per_domain"]
            def band(d):
                return f"{pd_[d]['band_rel_lo']:.2f}-{pd_[d]['band_rel_hi']:.2f}"
            inter = (f"{e['intersection_rel'][0]:.2f}-{e['intersection_rel'][1]:.2f}"
                     if e["intersection_rel"] else "empty")
            L.append(f"| {SHORT[slug]} | {band('athletes')} | {band('cities')} | {band('writers')} | "
                     f"{band('musicians')} | {inter} | {e['overlap_over_union']:.2f} | "
                     f"{e['best_rel_depth_spread']:.2f} |")
        L.append("")

    # ---- (d) per-head --------------------------------------------------
    L.append("## (d) Per-head attention-entropy analysis (E7, K-vs-F, prompt point)")
    L.append("")
    L.append("Heads ranked by the K-vs-F separability AUROC of their prompt-point attention "
             "entropy. Cross-domain consistency: Spearman of the full per-head AUROC vectors "
             "between domain pairs, and top-20 head-set overlap. A concentrated signal (few "
             "heads carry it across ALL domains) supports Ferrando's attribute-extraction-head "
             "story; a diffuse signal argues against it.")
    L.append("")
    for slug in SLUGS:
        e = res["d_per_head"][slug]
        L.append(f"### {SHORT[slug]} ({e['n_heads_total']} heads total)")
        L.append("")
        L.append(f"Mean pairwise Spearman of per-head AUROC vectors: **{e['spearman_mean']:.3f}** "
                 f"({', '.join(f'{k} {v:.2f}' for k, v in e['spearman_pairwise'].items())}).")
        L.append("")
        L.append(f"Heads in all four domains' top-20: **{len(e['heads_in_all_four_top20'])}** "
                 + (", ".join(f"L{h['layer']}H{h['head']} ({h['mean_auroc']:.3f})"
                              for h in e['heads_in_all_four_top20']) or "none") + ".")
        L.append("")
        L.append("Top-10 heads by mean K-vs-F AUROC across domains:")
        L.append("")
        L.append("| Layer | Head | rel.depth | mean AUROC | athletes | cities | writers | musicians |")
        L.append("|---|---|---|---|---|---|---|---|")
        for h in e["top10_by_mean_auroc"]:
            pdv = h["per_domain"]
            L.append(f"| {h['layer']} | {h['head']} | {h['rel_depth']:.2f} | {h['mean_auroc']:.3f} | "
                     f"{pdv['athletes']:.3f} | {pdv['cities']:.3f} | {pdv['writers']:.3f} | {pdv['musicians']:.3f} |")
        L.append("")
        conc_a = e["concentration"]["athletes"]
        L.append(f"Concentration (athletes): best head {conc_a['max']:.3f}, 10th-best "
                 f"{conc_a['top10_min']:.3f}, median {conc_a['median']:.3f}, "
                 f"{conc_a['frac_heads_ge_0.8'] * 100:.0f}% of heads >= 0.8, "
                 f"{conc_a['frac_heads_ge_0.9'] * 100:.0f}% >= 0.9.")
        L.append("")

    L.append(f"Wall-clock: {meta.get('wall_clock_s', '?')} s. Generated: {meta.get('generated_at')}.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Probe bootstrap CIs at the fixed best layer (K-vs-F, all domains)
# --------------------------------------------------------------------------

def probe_oof_proba(X: np.ndarray, labels: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Out-of-fold CV probabilities (mirror of the probe out-of-fold procedure)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    minority = int(np.bincount(labels).min())
    cv = StratifiedKFold(n_splits=min(5, minority), shuffle=True, random_state=seed)
    return cross_val_predict(_probe_pipeline(), X, labels, cv=cv, method="predict_proba")[:, 1]


def patch_probe_ci() -> None:
    """Add 95% bootstrap CIs to the K-vs-F probe cells of an EXISTING
    results/domain_generalization.json (all four domains), then re-render the md.

    Uses the fixed-best-layer convention: the CI bootstraps the
    out-of-fold 5-fold-CV probabilities at the FIXED best layer already
    selected (and stored) by section (a). 10k stratified resamples,
    numpy default_rng(0). No other number in the JSON is touched.
    """
    rng = np.random.default_rng(SEED)
    with open(JSON_PATH) as f:
        res = json.load(f)
    for slug in SLUGS:
        for domain in DOMAINS:
            cell = res["a_per_domain"][slug][domain]["K_vs_F"]["probe"]
            best_layer_id = int(cell["best_layer"])
            layer_ids, hidden, npz_cond = load_prompt_hidden(slug, domain)
            npz_mask, npz_labels = contrast_mask(npz_cond, "FABRICATED", "KNOWN")
            X = hidden[layer_ids.index(best_layer_id)][npz_mask]
            oof = probe_oof_proba(X, npz_labels)
            ci = bootstrap_auroc_ci(oof, npz_labels, N_BOOT, rng)
            # sanity: OOF separability must equal the stored CV AUROC
            oof_sep = float(max(ci["auroc"], 1.0 - ci["auroc"]))
            if abs(oof_sep - cell["auroc"]) > 1e-9:
                print(f"WARN {slug}/{domain}: OOF sep {oof_sep:.6f} != stored {cell['auroc']:.6f}")
            cell["ci95"] = ci["ci95"]
            cell["ci_n_boot"] = ci["n_boot"]
            cell["ci_note"] = ("CI bootstraps the out-of-fold CV probabilities at the "
                               "fixed best layer")
            print(f"{SHORT[slug]:14} {domain:10} probe {cell['auroc']:.3f}@L{best_layer_id} "
                  f"CI [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]", flush=True)
    res["meta"]["probe_ci_patch"] = {
        "n_boot": N_BOOT, "seed": SEED,
        "patched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "a_per_domain.*.*.K_vs_F.probe (all domains)",
    }
    with open(JSON_PATH, "w") as f:
        json.dump(_san(res), f, indent=2, ensure_ascii=False)
    MD_PATH.write_text(render_markdown(res))
    print(f"patched {JSON_PATH}")
    print(f"re-rendered {MD_PATH}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    results: dict = {"meta": {
        "seed": SEED, "n_boot": N_BOOT, "domains": DOMAINS,
        "models": SLUGS, "band_frac": BAND_FRAC,
        "separability": "max(auc, 1-auc)", "point": "prompt",
        "template_note": "cities uses 'Czym jest X?'; other domains use 'Kim jest X?'",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    print("(a) per-domain separability ...", flush=True)
    results["a_per_domain"] = section_a(rng)

    print("(b) transfer matrices ...", flush=True)
    packs = {slug: {d: _kf_pack(slug, d) for d in DOMAINS} for slug in SLUGS}
    results["b_transfer"] = section_b(packs)

    print("(c) layer-band stability ...", flush=True)
    results["c_layer_band"] = section_c()

    print("(d) per-head analysis ...", flush=True)
    results["d_per_head"] = section_d()

    results["meta"]["wall_clock_s"] = round(time.time() - t0, 1)
    results = _san(results)

    JSON_PATH.parent.mkdir(exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    MD_PATH.write_text(render_markdown(results))
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")
    print(f"total wall-clock: {results['meta']['wall_clock_s']} s")


if __name__ == "__main__":
    if "--patch-probe-ci" in sys.argv:
        patch_probe_ci()
    else:
        main()
