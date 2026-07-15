"""Lexical-naturalness baselines.

The real-vs-fabricated probe might
exploit a lexical shortcut (name plausibility) rather than entity familiarity.
Token-length was matched by construction, but character n-grams, name length,
and word structure were not. This script trains purely lexical classifiers on
the entity *string* and compares their real-vs-fabricated AUROC to the probe's
0.859-0.934 (Table 4a). If lexical models sit far below the probe, a naturalness
shortcut cannot explain the probe's performance.

Evaluated on the 320-entity v2 eval subset (240 real, 80 fabricated), the same
set as Table 4a. Char n-gram classifier scored 5-fold out-of-fold.

Outputs results/paper2_robustness/lexical_baselines.{json,md}.
"""
from __future__ import annotations
import os, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper2_robustness"
OUT.mkdir(parents=True, exist_ok=True)

TOKENIZERS = {
    "Bielik-1.5B": "speakleash/Bielik-1.5B-v3.0-Instruct",
    "Bielik-11B": "speakleash/Bielik-11B-v3.0-Instruct",
    "Gemma-12B": "google/gemma-4-12b-it",
    "Qwen-14B": "Qwen/Qwen3-14B",
}
# probe AUROC on the same target (Table 4a), per model, for reference
PROBE_REF = {"Bielik-11B": 0.934, "Gemma-12B": 0.871,
             "Qwen-14B": 0.859, "PLLuM-12B": 0.903}


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


def sep(y, feat):
    """Separability of a single scalar feature = max(AUROC, 1-AUROC)."""
    a = auroc(y, feat)
    return max(a, 1 - a)


def main():
    ev = pd.read_parquet(ROOT / "results/paper2_final/eval_subset.parquet")
    ev = ev[["entity", "kind", "domain"]].copy()
    y = (ev["kind"] == "fabricated").astype(int).to_numpy()
    names = ev["entity"].tolist()
    assert y.sum() == 80 and len(y) == 320, (int(y.sum()), len(y))

    feats = {}
    feats["char_length"] = np.array([len(n) for n in names], float)
    feats["word_count"] = np.array([len(n.split()) for n in names], float)
    feats["mean_word_len"] = np.array(
        [np.mean([len(w) for w in n.split()]) for n in names], float)

    from transformers import AutoTokenizer
    for short, hf in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(hf)
        feats[f"tokcount_{short}"] = np.array(
            [len(tok(n, add_special_tokens=False)["input_ids"]) for n in names],
            float)

    single = {k: round(sep(y, v), 3) for k, v in feats.items()}

    # char n-gram classifier, out-of-fold
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  min_df=2)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(pipe, names, y, cv=skf, method="predict_proba")[:, 1]
    char_ngram_auroc = round(auroc(y, oof), 3)

    # all lexical scalar features combined, out-of-fold
    X = np.column_stack(list(feats.values()))
    lr = LogisticRegression(max_iter=2000)
    oof2 = cross_val_predict(lr, X, y, cv=skf, method="predict_proba")[:, 1]
    lex_combined_auroc = round(auroc(y, oof2), 3)

    result = {
        "target": "real (0) vs fabricated (1), 320-entity eval subset",
        "single_feature_separability": single,
        "char_ngram_2_5_oof_auroc": char_ngram_auroc,
        "lexical_scalar_combined_oof_auroc": lex_combined_auroc,
        "probe_auroc_reference_table4a": PROBE_REF,
    }
    (OUT / "lexical_baselines.json").write_text(json.dumps(result, indent=2))

    md = ["# Lexical-naturalness baselines vs. the probe\n",
          "Real-vs-fabricated discrimination from the entity *string* alone, on the "
          "320-entity eval subset (240 real / 80 fabricated). If these sit far below "
          "the probe, a name-plausibility shortcut cannot explain it.\n",
          "## Single-feature separability = max(AUROC, 1-AUROC)\n",
          "| feature | separability |", "|---|---|"]
    for k, v in single.items():
        md.append(f"| {k} | {v} |")
    md += ["", "## Learned lexical classifiers (5-fold out-of-fold AUROC)\n",
           f"- **char n-gram (2-5) TF-IDF + logistic: {char_ngram_auroc}**",
           f"- combined lexical scalars (length/word/token-counts): "
           f"{lex_combined_auroc}", "",
           "## Probe reference (Table 4a, same target)\n",
           "| model | probe AUROC |", "|---|---|"]
    for k, v in PROBE_REF.items():
        md.append(f"| {k} | {v} |")
    md += ["",
           f"The strongest purely lexical model reaches AUROC "
           f"{max(char_ngram_auroc, lex_combined_auroc)}, versus the probe's "
           f"{min(PROBE_REF.values())}-{max(PROBE_REF.values())}. The gap is the "
           "margin not attributable to surface name form."]
    (OUT / "lexical_baselines.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
