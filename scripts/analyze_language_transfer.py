"""Paper-2 RQ1 (EN axis) analysis — language transfer of the familiarity signal.

Reads, for each of the six models with full artifacts (four Bielik v3.0-Instruct
sizes + two Gemma-4 sizes), the PL extraction (results/<slug>/ for athletes,
results/<slug>/domains/<d>/ for the other domains) and the EN extraction
(results/<slug>/domains/<d>_en/, produced by scripts/run_language_transfer.py:
same Polish entities, English question stems).

Computes and writes results/language_transfer.{json,md}:
  (1) within-language separability EN vs PL per model x domain (K-vs-F and
      K-vs-UR: best-layer dispersion + probe + first-token entropy — same
      conventions as analyze_gemma_control.py / domain_generalization.py).
  (2) CROSS-LANGUAGE probe transfer on the SAME entities: probe trained on
      PL K+F at PL's CV-best layer, scored zero-shot on EN K+F at the same
      layer index, and the reverse; reported raw and as a fraction of the
      TARGET language's within-language probe AUROC. Dispersion transfer
      (source language's best metric+layer read on the target language) as
      the secondary family. K-vs-F headline; K-vs-UR reported too because on
      Gemma the KNOWN lists carry the Bielik-design caveat.
  (3) pooled per model: does familiarity look language-independent
      (cross ~ within) or language-bound?
  (4) Gemma-specific check: Gemma is English-centric — do EN stems yield a
      STRONGER within-language signal than PL for the same Polish entities
      (esp. athletes/musicians, where the PL signal was weakest)?
  (5) honest caveats (template confounded with language; entities stay Polish).

Convention identical to analyze_gemma_control.py: prompt point, K-vs-F
headline, sep=max(auc,1-auc), numpy default_rng(0), sklearn CV random_state=0.

Usage:
    uv run python scripts/analyze_language_transfer.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
    "gemma-4-E4B-it",
    "gemma-4-12b-it",
]
SHORT = {
    "Bielik-1.5B-v3.0-Instruct": "Bielik-1.5B",
    "Bielik-4.5B-v3.0-Instruct": "Bielik-4.5B",
    "Bielik-Minitron-7B-v3.0-Instruct": "Bielik-7B",
    "Bielik-11B-v3.0-Instruct": "Bielik-11B",
    "gemma-4-E4B-it": "Gemma-4-E4B",
    "gemma-4-12b-it": "Gemma-4-12B",
}
GEMMA_SLUGS = ["gemma-4-E4B-it", "gemma-4-12b-it"]
DOMAINS = ["athletes", "cities", "writers", "musicians"]
LANGS = ("pl", "en")
CONTRASTS = [
    ("K_vs_F", "FABRICATED", "KNOWN"),
    ("K_vs_UR", "UNKNOWN_REAL", "KNOWN"),
]
SEED = 0
# cross/within fraction above which the representation is called
# language-independent in the pooled verdict.
LANG_INDEPENDENT_FRAC = 0.9

JSON_PATH = ROOT / "results" / "language_transfer.json"
MD_PATH = ROOT / "results" / "language_transfer.md"


def _results_dir(slug: str, domain: str, lang: str) -> Path:
    base = ROOT / "results" / slug
    if lang == "en":
        return base / "domains" / f"{domain}_en"
    return base if domain == "athletes" else base / "domains" / domain


def contrast_mask(conditions, pos, neg):
    mask = np.isin(conditions, [pos, neg])
    labels = (conditions[mask] == pos).astype(int)
    return mask, labels


def load_prompt_signals(slug, domain, lang):
    sig = pd.read_parquet(_results_dir(slug, domain, lang) / "signals.parquet")
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


def load_prompt_hidden(slug, domain, lang):
    z = np.load(_results_dir(slug, domain, lang) / "hidden_states.npz",
                allow_pickle=False)
    keys = sorted((k for k in z.files if k.startswith("prompt_layer_")),
                  key=lambda k: int(k.split("_")[-1]))
    layer_ids = [int(k.split("_")[-1]) for k in keys]
    hidden = [np.asarray(z[k], dtype=np.float32) for k in keys]
    return layer_ids, hidden, z["conditions"]


def sep(a):
    return np.maximum(a, 1.0 - a)


def auc_matrix(M, labels):
    R = rankdata(M, axis=1)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    s = R[:, labels == 1].sum(axis=1)
    return (s - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _probe_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def probe_sweep(hidden_list, labels):
    hb = {i: X for i, X in enumerate(hidden_list)}
    res = probe_auroc_per_layer(hb, labels, seed=SEED)
    aucs = np.array([res[i] for i in range(len(hidden_list))])
    return aucs, int(np.nanargmax(aucs))


def _pack(slug, domain, lang, pos, neg):
    """Everything a contrast needs for one (model, domain, language)."""
    layers, entities, conditions, mats, fte = load_prompt_signals(slug, domain, lang)
    layer_ids, hidden, npz_cond = load_prompt_hidden(slug, domain, lang)
    mask, labels = contrast_mask(conditions, pos, neg)
    npz_mask, npz_labels = contrast_mask(npz_cond, pos, neg)
    disp = {m: mats[m][:, mask] for m in ("ipr", "entropy")}
    hidden_c = [X[npz_mask] for X in hidden]
    probe_aucs, probe_best = probe_sweep(hidden_c, npz_labels)
    bm, bi, ba = None, None, -1.0
    for m in ("ipr", "entropy"):
        ma = sep(auc_matrix(disp[m], labels))
        k = int(np.argmax(ma))
        if ma[k] > ba:
            bm, bi, ba = m, k, float(ma[k])
    return {
        "layers": layers, "layer_ids": layer_ids,
        "disp": disp, "labels": labels,
        "hidden": hidden_c, "npz_labels": npz_labels,
        "probe_best_idx": probe_best, "probe_within": float(probe_aucs[probe_best]),
        "disp_best": (bm, bi), "disp_within": float(ba),
        "fte_auroc": float(sep(auc_matrix(fte[mask][None, :], labels))[0]),
    }


# --------------------------------------------------------------------------
# (1) within-language separability, EN vs PL
# --------------------------------------------------------------------------

def section_within(packs):
    out = {}
    for slug in SLUGS:
        out[slug] = {}
        for domain in DOMAINS:
            out[slug][domain] = {}
            for cname, _, _ in CONTRASTS:
                out[slug][domain][cname] = {}
                for lang in LANGS:
                    p = packs[(slug, domain, lang, cname)]
                    out[slug][domain][cname][lang] = {
                        "dispersion": round(p["disp_within"], 4),
                        "dispersion_metric": p["disp_best"][0],
                        "dispersion_layer": int(p["layers"][p["disp_best"][1]]),
                        "probe": round(p["probe_within"], 4),
                        "probe_layer": int(p["layer_ids"][p["probe_best_idx"]]),
                        "first_token_entropy": round(p["fte_auroc"], 4),
                    }
    return out


# --------------------------------------------------------------------------
# (2) cross-language transfer on the same entities
# --------------------------------------------------------------------------

def _transfer_one(src, tgt):
    """Probe: fit on src at src's best layer, score tgt at that layer.
    Dispersion: read src's best (metric, layer) on tgt."""
    ai = src["probe_best_idx"]
    clf = _probe_pipeline().fit(src["hidden"][ai], src["npz_labels"])
    proba = clf.predict_proba(tgt["hidden"][ai])[:, 1]
    a = roc_auc_score(tgt["npz_labels"], proba)
    probe_cross = float(max(a, 1 - a))
    am, al = src["disp_best"]
    disp_cross = float(sep(auc_matrix(tgt["disp"][am][al:al + 1], tgt["labels"]))[0])
    return probe_cross, disp_cross


def section_cross(packs):
    out = {}
    for slug in SLUGS:
        out[slug] = {}
        for domain in DOMAINS:
            out[slug][domain] = {}
            for cname, _, _ in CONTRASTS:
                pl = packs[(slug, domain, "pl", cname)]
                en = packs[(slug, domain, "en", cname)]
                p_pe, d_pe = _transfer_one(pl, en)  # PL -> EN
                p_ep, d_ep = _transfer_one(en, pl)  # EN -> PL
                out[slug][domain][cname] = {
                    "probe_pl_to_en": round(p_pe, 4),
                    "probe_en_to_pl": round(p_ep, 4),
                    "probe_frac_pl_to_en": round(p_pe / en["probe_within"], 4),
                    "probe_frac_en_to_pl": round(p_ep / pl["probe_within"], 4),
                    "dispersion_pl_to_en": round(d_pe, 4),
                    "dispersion_en_to_pl": round(d_ep, 4),
                }
    return out


# --------------------------------------------------------------------------
# (3) pooled per model + (4) Gemma EN-vs-PL
# --------------------------------------------------------------------------

def section_pooled(within, cross):
    out = {}
    for slug in SLUGS:
        pooled = {}
        for cname, _, _ in CONTRASTS:
            w_pl = np.mean([within[slug][d][cname]["pl"]["probe"] for d in DOMAINS])
            w_en = np.mean([within[slug][d][cname]["en"]["probe"] for d in DOMAINS])
            c_pe = np.mean([cross[slug][d][cname]["probe_pl_to_en"] for d in DOMAINS])
            c_ep = np.mean([cross[slug][d][cname]["probe_en_to_pl"] for d in DOMAINS])
            f_pe = np.mean([cross[slug][d][cname]["probe_frac_pl_to_en"] for d in DOMAINS])
            f_ep = np.mean([cross[slug][d][cname]["probe_frac_en_to_pl"] for d in DOMAINS])
            mean_frac = float((f_pe + f_ep) / 2)
            pooled[cname] = {
                "within_pl_probe": round(float(w_pl), 4),
                "within_en_probe": round(float(w_en), 4),
                "cross_pl_to_en_probe": round(float(c_pe), 4),
                "cross_en_to_pl_probe": round(float(c_ep), 4),
                "frac_pl_to_en": round(float(f_pe), 4),
                "frac_en_to_pl": round(float(f_ep), 4),
                "mean_cross_over_within": round(mean_frac, 4),
                "language_independent": bool(mean_frac >= LANG_INDEPENDENT_FRAC),
            }
        out[slug] = pooled
    return out


def section_gemma(within):
    out = {}
    for slug in GEMMA_SLUGS:
        rows = {}
        for domain in DOMAINS:
            per = {}
            for cname, _, _ in CONTRASTS:
                w = within[slug][domain][cname]
                per[cname] = {
                    "probe_pl": w["pl"]["probe"], "probe_en": w["en"]["probe"],
                    "probe_delta_en_minus_pl": round(w["en"]["probe"] - w["pl"]["probe"], 4),
                    "dispersion_pl": w["pl"]["dispersion"], "dispersion_en": w["en"]["dispersion"],
                    "dispersion_delta_en_minus_pl": round(
                        w["en"]["dispersion"] - w["pl"]["dispersion"], 4),
                }
            rows[domain] = per
        # pooled deltas
        for cname, _, _ in CONTRASTS:
            rows[f"mean_probe_delta_{cname}"] = round(float(np.mean(
                [rows[d][cname]["probe_delta_en_minus_pl"] for d in DOMAINS])), 4)
        out[slug] = rows
    return out


# --------------------------------------------------------------------------
# main + markdown
# --------------------------------------------------------------------------

def main():
    t0 = time.time()
    packs = {}
    for slug in SLUGS:
        for domain in DOMAINS:
            for lang in LANGS:
                for cname, pos, neg in CONTRASTS:
                    packs[(slug, domain, lang, cname)] = _pack(slug, domain, lang, pos, neg)
        print(f"loaded {slug}", flush=True)

    res = {"meta": {
        "seed": SEED, "models": SLUGS, "domains": DOMAINS,
        "en_templates": {"people": "Who is {entity}? Answer in one sentence.",
                         "places": "What is {entity}? Answer in one sentence."},
        "pl_templates": {"people": "Kim jest {entity}? Odpowiedz jednym zdaniem.",
                         "places": "Czym jest {entity}? Odpowiedz jednym zdaniem."},
        "convention": ("prompt point, sep=max(auc,1-auc), K-vs-F headline, "
                       "probe transfer at source-language CV-best layer, "
                       "fraction relative to target language's within probe"),
        "language_independent_threshold": LANG_INDEPENDENT_FRAC,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}
    res["within"] = section_within(packs)
    res["cross"] = section_cross(packs)
    res["pooled"] = section_pooled(res["within"], res["cross"])
    res["gemma_en_vs_pl"] = section_gemma(res["within"])
    res["meta"]["wall_clock_s"] = round(time.time() - t0, 1)

    JSON_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    MD_PATH.write_text(render_md(res))
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")


def render_md(res):
    L = []
    L.append("# Language transfer — same Polish entities, English question stems (paper-2 RQ1, EN axis)")
    L.append("")
    L.append("*Generated by `scripts/analyze_language_transfer.py` from the EN extractions "
             "of `scripts/run_language_transfer.py`. Six models (four Bielik v3.0-Instruct "
             "sizes, two Gemma-4 sizes), four domains, 126 entities each (42 KNOWN / 42 "
             "UNKNOWN_REAL / 42 FABRICATED).*")
    L.append("")
    L.append("## Design")
    L.append("")
    L.append("- Entities are UNCHANGED Polish strings; only the question stem switches "
             "language: PL *Kim/Czym jest {entity}? Odpowiedz jednym zdaniem.* -> EN "
             "*Who/What is {entity}? Answer in one sentence.*")
    L.append("- Forward-only extraction at the prompt point (same conventions as the "
             "neutral-template control): dispersion (IPR / Shannon entropy over MLP "
             "act_fn vectors, best layer), residual-stream logistic probe (5-fold CV, "
             "best layer), first-token entropy baseline. sep = max(AUROC, 1-AUROC).")
    L.append("- Cross-language transfer: probe fit on ALL of the source language's K+F "
             "(or K+UR) at the source's CV-best layer, scored zero-shot on the target "
             "language at the SAME layer index, on the SAME entities. Reported raw and "
             "as a fraction of the target language's within-language probe AUROC.")
    L.append("")

    L.append("## (1) Within-language separability — EN vs PL")
    L.append("")
    for cname, title in (("K_vs_F", "KNOWN vs FABRICATED (headline familiarity)"),
                         ("K_vs_UR", "KNOWN vs UNKNOWN-REAL (fabricated-string-free)")):
        L.append(f"### {title}")
        L.append("")
        L.append("| Model | Domain | PL dispersion | PL probe | EN dispersion | EN probe | EN-PL probe delta |")
        L.append("|---|---|---|---|---|---|---|")
        for slug in SLUGS:
            for d in DOMAINS:
                w = res["within"][slug][d][cname]
                pl, en = w["pl"], w["en"]
                delta = en["probe"] - pl["probe"]
                L.append(
                    f"| {SHORT[slug]} | {d} | {pl['dispersion']:.3f}@L{pl['dispersion_layer']} "
                    f"({pl['dispersion_metric']}) | {pl['probe']:.3f}@L{pl['probe_layer']} | "
                    f"{en['dispersion']:.3f}@L{en['dispersion_layer']} ({en['dispersion_metric']}) | "
                    f"{en['probe']:.3f}@L{en['probe_layer']} | {delta:+.3f} |")
        L.append("")

    L.append("## (2) Cross-language probe transfer (same entities)")
    L.append("")
    for cname in ("K_vs_F", "K_vs_UR"):
        L.append(f"### {cname}")
        L.append("")
        L.append("| Model | Domain | PL->EN | frac of EN-within | EN->PL | frac of PL-within | disp PL->EN | disp EN->PL |")
        L.append("|---|---|---|---|---|---|---|---|")
        for slug in SLUGS:
            for d in DOMAINS:
                c = res["cross"][slug][d][cname]
                L.append(
                    f"| {SHORT[slug]} | {d} | {c['probe_pl_to_en']:.3f} | "
                    f"{c['probe_frac_pl_to_en']:.3f} | {c['probe_en_to_pl']:.3f} | "
                    f"{c['probe_frac_en_to_pl']:.3f} | {c['dispersion_pl_to_en']:.3f} | "
                    f"{c['dispersion_en_to_pl']:.3f} |")
        L.append("")

    L.append("## (3) Pooled per model — is the familiarity representation language-independent?")
    L.append("")
    L.append("Probe AUROC pooled (mean) over the four domains.")
    L.append("")
    for cname in ("K_vs_F", "K_vs_UR"):
        L.append(f"### {cname}")
        L.append("")
        L.append("| Model | within PL | within EN | cross PL->EN | cross EN->PL | mean cross/within | verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for slug in SLUGS:
            p = res["pooled"][slug][cname]
            v = "language-independent" if p["language_independent"] else "language-bound"
            L.append(
                f"| {SHORT[slug]} | {p['within_pl_probe']:.3f} | {p['within_en_probe']:.3f} | "
                f"{p['cross_pl_to_en_probe']:.3f} | {p['cross_en_to_pl_probe']:.3f} | "
                f"{p['mean_cross_over_within']:.3f} | {v} |")
        L.append("")

    L.append("## (4) Gemma check — English-centric family, English stems")
    L.append("")
    L.append("Prediction: Gemma (English-centric) should show a stronger WITHIN-language "
             "signal under EN stems than PL stems for the same Polish entities, especially "
             "in the domains where the PL signal was weakest (athletes, musicians).")
    L.append("")
    L.append("| Model | Domain | K-vs-F probe PL | K-vs-F probe EN | delta | K-vs-UR probe PL | K-vs-UR probe EN | delta |")
    L.append("|---|---|---|---|---|---|---|---|")
    for slug in GEMMA_SLUGS:
        g = res["gemma_en_vs_pl"][slug]
        for d in DOMAINS:
            kf, kur = g[d]["K_vs_F"], g[d]["K_vs_UR"]
            L.append(
                f"| {SHORT[slug]} | {d} | {kf['probe_pl']:.3f} | {kf['probe_en']:.3f} | "
                f"{kf['probe_delta_en_minus_pl']:+.3f} | {kur['probe_pl']:.3f} | "
                f"{kur['probe_en']:.3f} | {kur['probe_delta_en_minus_pl']:+.3f} |")
    L.append("")
    for slug in GEMMA_SLUGS:
        g = res["gemma_en_vs_pl"][slug]
        L.append(f"- {SHORT[slug]}: mean probe delta (EN-PL) — K-vs-F "
                 f"{g['mean_probe_delta_K_vs_F']:+.3f}, K-vs-UR "
                 f"{g['mean_probe_delta_K_vs_UR']:+.3f}.")
    L.append("")

    L.append("## (5) Caveats")
    L.append("")
    L.append("- **Template is confounded with language.** The EN stems are translations, "
             "so the language switch also changes the surface template. The neutral-template "
             "control (cities_neutral / writers_neutral) showed the Bielik familiarity signal "
             "is robust to a template swap within Polish, which bounds — but does not "
             "eliminate — this confound; a same-language template control at this scale is "
             "the precedent we lean on.")
    L.append("- **Entity strings remain Polish in both conditions.** This is by design "
             "(the question is whether the QUESTION language matters for the same entity "
             "representation), but it means the EN prompts are code-mixed rather than fully "
             "English; entity tokenization is identical across the language pair.")
    L.append("- **Layer pairing.** Cross-language scoring reuses the source language's best "
             "layer index; if the two languages peaked at different depths, the fraction "
             "understates transferability at the target's own best layer.")
    L.append("- **KNOWN-list design caveat (Gemma).** KNOWN entities were selected for "
             "Bielik; for Gemma the K-vs-UR contrast remains the fabricated-string-free "
             "control of record (see results/gemma_control.md).")
    L.append("")
    m = res["meta"]
    L.append(f"Wall-clock: {m.get('wall_clock_s', '?')} s. Generated: {m['generated_at']}.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
