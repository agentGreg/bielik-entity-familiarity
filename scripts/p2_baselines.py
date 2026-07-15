"""Paper-2 final phase, step 3: literature baselines vs our one-pass signals.

Scores every eval-subset entity per model with:

Generation-time literature baselines (from the captured shared pass):
  * D2HScore (Ding et al. 2025)  — training-free. ScoreDispersion = mean over
    layers of intra-layer dispersion; ScoreDrift = mean over adjacent layers
    of attention-guided core-representation drift (top-40% tokens); both
    min-max normalized within the per-model eval pool, equally weighted.
    Direction per paper: LOW D2HScore => hallucination, so risk = -D2HScore.
  * EigenTrack (Ettori et al. 2025) — supervised GRU over per-step spectral
    feature sequences. Adaptation: trained with 5-fold stratified CV on OUR
    contrast labels (the paper trains supervised on HaluEval-style labeled
    data); AUROC uses out-of-fold probabilities.
  * MIND (Su et al. 2024) — MLP (1 hidden ReLU layer) on the last generated
    token's last-layer hidden state (the paper's chosen feature). Adaptation:
    the paper's unsupervised Wikipedia-continuation auto-labeling pipeline is
    NOT reproducible in our single-question PL setting; the classifier is
    trained with 5-fold CV on our labels instead (out-of-fold AUROC).

Multi-sample baselines:
  * Discrete semantic entropy (Kuhn/Farquhar) over the 5 sampled answers,
    clustered by claude-sonnet-5 (same protocol as scripts/semantic_entropy.py).

Our one-pass prompt-point signals (from the v2 campaign artifacts, SAME
entities, no new GPU work):
  * dispersion best-layer metric (per-(model,domain) best point/metric/layer
    from results/v2_campaign.json; orientation fixed from the campaign decile
    curve, z-scored per domain before pooling),
  * calibrated familiarity-probe risk (logistic probe on prompt-point residual
    states, trained on NON-EVAL top-3-deciles-vs-fabricated rows per model,
    Platt-calibrated, per scripts/train_risk_probe.py conventions),
  * first-token entropy (zero-cost baseline).

Contrasts:
  (a) fabricated-vs-real over the full 320-entity pool per model;
  (b) behavioral: REAL entities only, majority-correct (>=3/5 strict-judge
      correct) vs not, labels from scripts/p2_calibration.py.

Outputs: results/paper2_final/{baseline_scores.parquet, baselines_auroc.json,
se_checkpoint.jsonl, risk_probe_meta.json}.

Usage:
    set -a; source .env; set +a
    uv run python scripts/p2_baselines.py [--skip-se]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_generate_eval_subset import entity_uid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "paper2_final"
GEN_ROOT = OUT_ROOT / "gen"
SE_CHECKPOINT = OUT_ROOT / "se_checkpoint.jsonl"

DOMAINS = ("athletes", "cities", "writers", "musicians")
SLUG_TO_SHORT = {
    "Bielik-11B-v3.0-Instruct": "Bielik-11B",
    "gemma-4-12b-it": "gemma-4-12b",
    "Qwen3-14B": "Qwen3-14B",
    "PLLuM-12B-instruct-2512": "PLLuM-12B",
}
SLUGS = tuple(SLUG_TO_SHORT)

CLUSTER_MODEL = "claude-sonnet-5"
SEED = 0
TOP_DECILES = (7, 8, 9)  # probe positive-familiarity training deciles
SE_WORKERS = 8           # concurrent clustering calls (lock-guarded checkpoint)

QUESTION_STEM = {
    "athletes": "Kim jest {entity}?", "writers": "Kim jest {entity}?",
    "musicians": "Kim jest {entity}?", "cities": "Czym jest {entity}?",
}

CLUSTER_PROMPT_TEMPLATE = """Pytanie: "{question}"

Poniżej jest 5 ponumerowanych odpowiedzi (po polsku) na to pytanie:
{numbered_answers}

Zadanie: pogrupuj te odpowiedzi w klastry znaczeniowe. Dwie odpowiedzi
należą do tego samego klastra wtedy i tylko wtedy, gdy przekazują to samo
główne twierdzenie o podmiocie pytania. Zignoruj różnice stylistyczne,
szyk zdania i drobne szczegóły (dodatkowe drugorzędne fakty), które nie
zmieniają głównego twierdzenia. Jeśli odpowiedzi podają sprzeczne lub różne
główne twierdzenia (np. różny zawód, różna tożsamość, jedna odpowiedź to
odmowa/brak wiedzy a inna podaje konkretną osobę), umieść je w osobnych
klastrach.

Zwróć WYŁĄCZNIE JSON w formacie:
{{"clusters": [[1,3],[2],[4,5]]}}

gdzie liczby to 1-indeksowane numery odpowiedzi. Każdy numer od 1 do 5
musi wystąpić dokładnie raz, w dokładnie jednym klastrze. Nie dodawaj
żadnego innego tekstu poza tym JSON-em."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable).
# ---------------------------------------------------------------------------
def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def d2h_score(disp_means: np.ndarray, drift_means: np.ndarray,
              w1: float = 0.5, w2: float = 0.5) -> np.ndarray:
    """Pool-level D2HScore fusion (paper eq. 10, w1=w2=0.5)."""
    return w1 * minmax_normalize(disp_means) + w2 * minmax_normalize(drift_means)


def discrete_semantic_entropy(clusters: list[list[int]], n: int = 5) -> float:
    se = 0.0
    for c in clusters:
        p = len(c) / n
        se -= p * math.log(p)
    return se


def validate_clusters(clusters, n: int = 5) -> list[list[int]] | None:
    if not isinstance(clusters, list) or not clusters:
        return None
    seen: list[int] = []
    normalized: list[list[int]] = []
    for c in clusters:
        if not isinstance(c, list) or not c:
            return None
        try:
            c_int = [int(x) for x in c]
        except (TypeError, ValueError):
            return None
        normalized.append(c_int)
        seen.extend(c_int)
    if sorted(seen) != list(range(1, n + 1)):
        return None
    return normalized


def extract_json_obj(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "clusters" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "clusters" in obj:
                return obj
        except json.JSONDecodeError:
            return None
    return None


def auroc_with_ci(scores: np.ndarray, labels: np.ndarray,
                  n_boot: int = 1000, seed: int = SEED) -> dict:
    """Raw AUROC (fixed orientation: higher score = predicted positive)."""
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return {"auroc": float("nan"), "ci95": [float("nan")] * 2,
                "n": int(len(labels)), "n_pos": int(labels.sum())}
    a = roc_auc_score(labels, scores)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots
              else (float("nan"), float("nan")))
    return {"auroc": round(float(a), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(len(labels)), "n_pos": int(labels.sum())}


# ---------------------------------------------------------------------------
# Semantic entropy via Claude clustering (checkpointed).
# ---------------------------------------------------------------------------
def call_with_retry(fn, tries: int = 8, base_delay: float = 2.0, sleep=time.sleep):
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < tries - 1:
                sleep(base_delay * 2**attempt)
    raise last_exc


def cluster_answers(client, question: str, answers: list[str]) -> list[list[int]]:
    numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = CLUSTER_PROMPT_TEMPLATE.format(question=question, numbered_answers=numbered)

    def _do():
        msg = client.messages.create(
            model=CLUSTER_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}])
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise ValueError(f"no text block: {msg.content!r}")
        obj = extract_json_obj(text)
        if obj is None:
            raise ValueError(f"unparseable clustering response: {text!r}")
        clusters = validate_clusters(obj.get("clusters"), n=len(answers))
        if clusters is None:
            raise ValueError(f"invalid partition: {obj!r}")
        return clusters
    return call_with_retry(_do)


def compute_semantic_entropy(records: dict[tuple, dict]) -> dict[tuple, float]:
    """records: (slug, qid) -> answers.jsonl record. Returns SE per key."""
    import anthropic
    client = anthropic.Anthropic()
    done: dict[tuple, dict] = {}
    if SE_CHECKPOINT.exists():
        with SE_CHECKPOINT.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[(rec["slug"], rec["uid"])] = rec
    out: dict[tuple, float] = {}
    lock = threading.Lock()
    pending = []
    for (slug, uid), rec in records.items():
        if (slug, uid) in done:
            out[(slug, uid)] = float(done[(slug, uid)]["se"])
        else:
            pending.append((slug, uid, rec))

    counter = {"n": 0}

    def _one(slug: str, uid: str, rec: dict) -> None:
        question = QUESTION_STEM[rec["domain"]].format(entity=rec["entity"])
        try:
            clusters = cluster_answers(client, question, rec["answers"])
            fallback = False
        except Exception as exc:  # noqa: BLE001 - exhausted retries
            print(f"  SE FAIL {slug}/{rec['entity']}: {exc} -> all-distinct",
                  flush=True)
            clusters = [[i] for i in range(1, 6)]
            fallback = True
        se = discrete_semantic_entropy(clusters)
        crec = {"slug": slug, "uid": uid, "entity": rec["entity"],
                "clusters": clusters, "se": se, "fallback": fallback}
        with lock:
            with SE_CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(crec, ensure_ascii=False) + "\n")
                f.flush()
            out[(slug, uid)] = se
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                print(f"  [se] {counter['n']}/{len(pending)} clustering calls...",
                      flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=SE_WORKERS) as pool:
            futs = [pool.submit(_one, s, u, r) for s, u, r in pending]
            for f in futs:
                f.result()
    print(f"[se] {len(pending)} API calls this run", flush=True)
    return out


# ---------------------------------------------------------------------------
# Supervised baselines: EigenTrack GRU + MIND MLP (out-of-fold CV).
# ---------------------------------------------------------------------------
def gru_cv_probs(sequences: list[np.ndarray], labels: np.ndarray,
                 n_splits: int = 5, seed: int = SEED, hidden: int = 16,
                 epochs: int = 150, lr: float = 0.01) -> np.ndarray:
    """Out-of-fold P(positive) from a GRU over feature sequences (CPU torch).

    EigenTrack classifier: linear projection -> single GRU layer -> binary
    head on the final hidden state. Features standardized on train folds.
    """
    import torch
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=np.float32)
    n = len(sequences)
    n_feat = max((s.shape[1] for s in sequences if s.size), default=16)
    seqs = [s if s.size else np.zeros((1, n_feat), dtype=np.float32)
            for s in sequences]
    lengths = np.array([len(s) for s in seqs])
    max_len = int(lengths.max())
    X = np.zeros((n, max_len, n_feat), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s

    probs = np.full(n, np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    class GRUClf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(n_feat, hidden)
            self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 1)

        def forward(self, x, lens):
            h = torch.relu(self.proj(x))
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                h, lens, batch_first=True, enforce_sorted=False)
            _, hn = self.gru(packed)
            return self.head(hn[-1]).squeeze(-1)

    for fold, (tr, te) in enumerate(skf.split(X, labels.astype(int))):
        torch.manual_seed(seed + fold)
        # Standardize per feature over train-fold steps (masked).
        flat = np.concatenate([X[i, :lengths[i]] for i in tr])
        mu, sd = flat.mean(0), flat.std(0) + 1e-8
        Xn = (X - mu) / sd
        for i in range(n):  # keep padding zeros
            Xn[i, lengths[i]:] = 0.0
        xt = torch.tensor(Xn[tr]); yt = torch.tensor(labels[tr])
        lt = torch.tensor(lengths[tr], dtype=torch.int64)
        xe = torch.tensor(Xn[te]); le = torch.tensor(lengths[te], dtype=torch.int64)
        model = GRUClf()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(model(xt, lt), yt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs[te] = torch.sigmoid(model(xe, le)).numpy()
    return probs


def mlp_cv_probs(X: np.ndarray, labels: np.ndarray, n_splits: int = 5,
                 seed: int = SEED) -> np.ndarray:
    """Out-of-fold P(positive) from the MIND MLP (1 hidden ReLU layer).

    Nested CV: the L2 strength (alpha) is selected on inner 3-fold grid
    search within each outer train fold — our sample (n~240-320) is far
    smaller than MIND's 5k training set, so regularization must be tuned
    rather than fixed to be fair to the baseline.
    """
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                         cross_val_predict)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                      max_iter=1000, random_state=seed))
    grid = GridSearchCV(
        pipe, {"mlpclassifier__alpha": [0.1, 1.0, 10.0]},
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="roc_auc", n_jobs=1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(grid, X, labels.astype(int), cv=cv,
                             method="predict_proba")[:, 1]


# ---------------------------------------------------------------------------
# Our signals from campaign artifacts.
# ---------------------------------------------------------------------------
def load_campaign_analysis() -> dict:
    return json.loads((ROOT / "results" / "v2_campaign.json").read_text())["analysis"]


def dispersion_risk_for_slug(slug: str, subset: pd.DataFrame,
                             analysis: dict) -> pd.Series:
    """Per-(model,domain) best-layer dispersion metric as a risk score.

    Point/metric/layer choices come from the campaign analysis (computed on
    the FULL 360-row v2 set). Orientation is re-derived from the RAW metric
    (the campaign's stored decile curves are sign-normalized): familiarity
    points from the fabricated mean to the decile-9 mean — the same anchor
    the campaign used. risk = -familiarity, z-scored per domain before
    pooling (per-domain best layers/metrics live on different scales).
    """
    short = SLUG_TO_SHORT[slug]
    out = {}
    for domain in DOMAINS:
        cfg = analysis[short][domain]["dispersion"]
        point, metric, layer = cfg["best_point"], cfg["best_metric"], cfg["best_layer"]
        sig = pd.read_parquet(
            ROOT / "results" / slug / "v2" / domain / "signals.parquet")
        sig = sig[(sig["layer"] == layer) & (sig["point"] == point)]
        vals = sig.set_index("entity")[metric]
        lab = pd.read_parquet(
            ROOT / "data" / slug / "v2" / domain / "labeled.parquet")
        v_all = vals.reindex(lab["entity"]).to_numpy(dtype=np.float64)
        d9_mean = v_all[(lab["decile"] == 9).to_numpy()].mean()
        fab_mean = v_all[(lab["kind"] != "real").to_numpy()].mean()
        familiar_high = d9_mean > fab_mean
        ents = subset[subset["domain"] == domain]
        v = vals.reindex(ents["entity"]).to_numpy(dtype=np.float64)
        fam = v if familiar_high else -v
        z = (fam - np.nanmean(fam)) / (np.nanstd(fam) + 1e-12)
        for uid, r in zip(ents["uid"], -z):
            out[uid] = r
    return pd.Series(out, name="risk_dispersion")


def fte_for_slug(slug: str, subset: pd.DataFrame) -> pd.Series:
    out = {}
    for domain in DOMAINS:
        sig = pd.read_parquet(
            ROOT / "results" / slug / "v2" / domain / "signals.parquet")
        one = sig[(sig["layer"] == sig["layer"].iloc[0])
                  & (sig["point"] == "prompt")]
        vals = one.set_index("entity")["first_token_entropy"]
        ents = subset[subset["domain"] == domain]
        for uid, v in zip(ents["uid"], vals.reindex(ents["entity"])):
            out[uid] = float(v)
    return pd.Series(out, name="risk_fte")


def train_probe_for_slug(slug: str, subset: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Calibrated familiarity-probe risk, trained on NON-EVAL v2 rows.

    Training pool per domain: real deciles 7-9 (label 0) + fabricated
    (label 1), excluding every eval-subset qid. Best prompt-point residual
    layer by 5-fold CV AUROC on the pooled task; Platt/sigmoid calibration on
    out-of-fold decision scores; final probe fit on all training rows
    (train_risk_probe.py conventions). Returns calibrated P(risk) for eval
    entities + metadata.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    eval_uids = set(subset["uid"])
    X_by_layer_train: dict[int, list[np.ndarray]] = {}
    y_train: list[int] = []
    X_by_layer_eval: dict[int, list[np.ndarray]] = {}
    eval_uid_order: list[str] = []

    for domain in DOMAINS:
        lab = pd.read_parquet(
            ROOT / "data" / slug / "v2" / domain / "labeled.parquet")
        npz = np.load(ROOT / "results" / slug / "v2" / domain / "hidden_states.npz")
        layers = sorted(int(k.split("_")[-1]) for k in npz.files
                        if k.startswith("prompt_layer_"))
        lab_uid = np.array([entity_uid(domain, e, q)
                            for e, q in zip(lab["entity"], lab["qid"])])
        is_eval = np.isin(lab_uid, list(eval_uids))
        is_fab = (lab["kind"] != "real").to_numpy()
        is_top = lab["decile"].isin(TOP_DECILES).to_numpy()
        train_mask = ~is_eval & (is_fab | is_top)
        dom_eval_mask = is_eval.copy()
        for l in layers:
            arr = npz[f"prompt_layer_{l}"]
            X_by_layer_train.setdefault(l, []).append(arr[train_mask])
            X_by_layer_eval.setdefault(l, []).append(arr[dom_eval_mask])
        y_train.extend(is_fab[train_mask].astype(int).tolist())
        eval_uid_order.extend(lab_uid[dom_eval_mask].tolist())

    y = np.array(y_train)
    layers = sorted(X_by_layer_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_layer, best_auc = None, -1.0
    for l in layers:
        X = np.concatenate(X_by_layer_train[l])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000))
        scores = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
        a = roc_auc_score(y, scores)
        if a > best_auc:
            best_auc, best_layer = a, l

    X = np.concatenate(X_by_layer_train[best_layer])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof_scores = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
    # Platt scaling as in train_risk_probe._fit_platt: 1-D logistic on the
    # out-of-fold decision scores (honest calibration, no leakage).
    platt = LogisticRegression(max_iter=1000)
    platt.fit(oof_scores.reshape(-1, 1), y)
    clf.fit(X, y)

    Xe = np.concatenate(X_by_layer_eval[best_layer])
    p_eval = platt.predict_proba(clf.decision_function(Xe).reshape(-1, 1))[:, 1]
    meta = {"best_layer": int(best_layer), "train_cv_auroc": round(float(best_auc), 4),
            "n_train": int(len(y)), "n_train_pos": int(y.sum())}
    return pd.Series(dict(zip(eval_uid_order, p_eval)), name="risk_probe"), meta


# ---------------------------------------------------------------------------
# Main assembly.
# ---------------------------------------------------------------------------
def load_captures(slug: str) -> dict[str, dict]:
    caps: dict[str, dict] = {}
    for domain in DOMAINS:
        dom_dir = GEN_ROOT / slug / domain
        with (dom_dir / "answers.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                z = np.load(dom_dir / "cap" / f"{rec['uid']}.npz")
                caps[rec["uid"]] = {
                    "rec": rec,
                    "disp_mean": float(z["disp"].mean()),
                    "drift_mean": float(z["drift"].mean()),
                    "eig_seq": z["eig_seq"].astype(np.float32),
                    "mind_last": z["mind_last"].astype(np.float32),
                }
    return caps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-se", action="store_true")
    ap.add_argument("--models", type=str, default=None)
    args = ap.parse_args()
    slugs = tuple(args.models.split(",")) if args.models else SLUGS

    subset = pd.read_parquet(OUT_ROOT / "eval_subset.parquet")
    beh_path = OUT_ROOT / "behavioral_labels.parquet"
    beh = pd.read_parquet(beh_path) if beh_path.exists() else None
    analysis = load_campaign_analysis()

    all_rows: list[pd.DataFrame] = []
    results: dict = {"per_model": {}, "seed": SEED}
    probe_meta: dict = {}

    for slug in slugs:
        print(f"=== {slug} ===", flush=True)
        caps = load_captures(slug)
        uids = [u for u in subset["uid"] if u in caps]
        if len(uids) != len(subset):
            print(f"  WARNING: {len(uids)}/{len(subset)} captured", flush=True)
        df = pd.DataFrame({"uid": uids})
        df["slug"] = slug
        for col in ("domain", "kind", "decile", "entity"):
            df[col] = subset.set_index("uid")[col].reindex(uids).to_numpy()
        df["label_fab"] = (df["kind"] != "real").astype(int)

        # D2HScore (pool normalization within model, per paper).
        disp = np.array([caps[u]["disp_mean"] for u in uids])
        drift = np.array([caps[u]["drift_mean"] for u in uids])
        d2h = d2h_score(disp, drift)
        df["d2h_dispersion"] = disp
        df["d2h_drift"] = drift
        df["d2h"] = d2h
        df["risk_d2h"] = -d2h

        # Our signals.
        df["risk_dispersion"] = dispersion_risk_for_slug(
            slug, subset, analysis).reindex(uids).to_numpy()
        df["risk_fte"] = fte_for_slug(slug, subset).reindex(uids).to_numpy()
        probe_series, meta = train_probe_for_slug(slug, subset)
        probe_meta[slug] = meta
        df["risk_probe"] = probe_series.reindex(uids).to_numpy()

        # Semantic entropy.
        if not args.skip_se:
            records = {(slug, u): caps[u]["rec"] for u in uids}
            se = compute_semantic_entropy(records)
            df["risk_se"] = [se[(slug, u)] for u in uids]
        else:
            df["risk_se"] = np.nan

        # Supervised baselines, contrast (a): fabricated vs real.
        seqs = [caps[u]["eig_seq"] for u in uids]
        mindX = np.stack([caps[u]["mind_last"] for u in uids])
        y_fab = df["label_fab"].to_numpy()
        print("  training EigenTrack GRU (contrast a)...", flush=True)
        df["eigentrack_p_fab"] = gru_cv_probs(seqs, y_fab)
        print("  training MIND MLP (contrast a)...", flush=True)
        df["mind_p_fab"] = mlp_cv_probs(mindX, y_fab)

        # Contrast (b): behavioral, real entities only.
        df["label_beh"] = np.nan
        if beh is not None and slug in set(beh["slug"]):
            b = beh[beh["slug"] == slug].set_index("uid")
            real_mask = df["kind"] == "real"
            df.loc[real_mask, "label_beh"] = (
                1 - b["majority_correct"].reindex(df.loc[real_mask, "uid"])
            ).to_numpy()
            rm = real_mask.to_numpy()
            y_beh = df.loc[rm, "label_beh"].to_numpy().astype(int)
            if 0 < y_beh.sum() < len(y_beh):
                print("  training EigenTrack GRU (contrast b)...", flush=True)
                p = np.full(len(df), np.nan)
                p[rm] = gru_cv_probs([s for s, m in zip(seqs, rm) if m], y_beh)
                df["eigentrack_p_beh"] = p
                print("  training MIND MLP (contrast b)...", flush=True)
                p = np.full(len(df), np.nan)
                p[rm] = mlp_cv_probs(mindX[rm], y_beh)
                df["mind_p_beh"] = p
            else:
                df["eigentrack_p_beh"] = np.nan
                df["mind_p_beh"] = np.nan
        else:
            df["eigentrack_p_beh"] = np.nan
            df["mind_p_beh"] = np.nan

        # AUROC tables.
        methods_a = {
            "d2hscore": df["risk_d2h"], "eigentrack": df["eigentrack_p_fab"],
            "mind": df["mind_p_fab"], "dispersion_best_layer": df["risk_dispersion"],
            "probe_calibrated": df["risk_probe"], "first_token_entropy": df["risk_fte"],
            "semantic_entropy": df["risk_se"],
        }
        contrast_a = {name: auroc_with_ci(s.to_numpy(), y_fab)
                      for name, s in methods_a.items()}
        contrast_b = {}
        if df["label_beh"].notna().any():
            rm = (df["kind"] == "real").to_numpy()
            y_b = df.loc[rm, "label_beh"].to_numpy().astype(int)
            methods_b = {
                "d2hscore": df.loc[rm, "risk_d2h"],
                "eigentrack": df.loc[rm, "eigentrack_p_beh"],
                "mind": df.loc[rm, "mind_p_beh"],
                "dispersion_best_layer": df.loc[rm, "risk_dispersion"],
                "probe_calibrated": df.loc[rm, "risk_probe"],
                "first_token_entropy": df.loc[rm, "risk_fte"],
                "semantic_entropy": df.loc[rm, "risk_se"],
            }
            contrast_b = {name: auroc_with_ci(s.to_numpy(), y_b)
                          for name, s in methods_b.items()}
        results["per_model"][slug] = {
            "contrast_a_fab_vs_real": contrast_a,
            "contrast_b_behavioral_real_only": contrast_b,
        }
        all_rows.append(df)
        print(f"  contrast (a) AUROCs: "
              f"{ {k: v['auroc'] for k, v in contrast_a.items()} }", flush=True)

    scores = pd.concat(all_rows, ignore_index=True)
    scores.to_parquet(OUT_ROOT / "baseline_scores.parquet")
    results["probe_meta"] = probe_meta
    with (OUT_ROOT / "baselines_auroc.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (OUT_ROOT / "risk_probe_meta.json").open("w", encoding="utf-8") as f:
        json.dump(probe_meta, f, indent=2)
    print(f"wrote {OUT_ROOT / 'baseline_scores.parquet'} ({len(scores)} rows) "
          f"and baselines_auroc.json", flush=True)


if __name__ == "__main__":
    main()
