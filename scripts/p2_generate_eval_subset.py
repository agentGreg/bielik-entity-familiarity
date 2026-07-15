"""Paper-2 final phase, step 1: eval subset + shared generation pass (GPU).

Builds a stratified eval subset of dataset v2 (per domain: 6 real entities per
decile 0-9 = 60, plus 20 fabricated = 80; 320 per model over 4 domains) and,
for each of the 4 final-phase models, samples 5 Polish answers per entity
(T=0.7, max 64 new tokens) plus ONE capture forward pass over prompt+answer#0
that records everything the literature baselines need:

  * D2HScore (Ding et al. 2025): per-layer intra-layer dispersion D_l over
    response tokens, and per-layer-pair drift delta_l where the layer "core"
    is the mean hidden state of the top-40% response tokens by attention
    received from the final response token (head-averaged) — computed on the
    fly, only the (L,) / (L-1,) vectors are stored.
  * EigenTrack (Ettori et al. 2025): per-generation-step spectral feature
    vectors F_t from a sliding-window (N=8) matrix of concatenated hidden
    states from ~8 evenly sampled layers; 16 features per step (leading
    eigenvalues, spectral gaps, spectral entropy/variance, effective rank,
    KL + Wasserstein divergence from the Marchenko-Pastur reference).
    Adaptation: window N=8 (paper knee is 25-50 tokens, but our answers are
    single sentences <=64 tokens; their ablation shows shorter windows help).
  * MIND (Su et al. 2024): last-layer hidden state of the last response token
    (the paper's chosen classifier input) + mean over response tokens.

Checkpointing is entity-keyed: gen/<slug>/<domain>/cap/<qid>.npz plus a
record in gen/<slug>/<domain>/answers.jsonl. Re-runs skip completed entities.

Usage:
    uv run python scripts/p2_generate_eval_subset.py --build-subset
    uv run python scripts/p2_generate_eval_subset.py --model speakleash/Bielik-11B-v3.0-Instruct
    uv run python scripts/p2_generate_eval_subset.py --all   # subprocess per model

Ownership: writes only under results/paper2_final/. Reads data/v2 and
bielik_hallu templates. Seeds: numpy default_rng(0) for the subset; per-sample
generation seeds derived from crc32(qid|sample_idx) for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_V2 = ROOT / "data" / "v2"
OUT_ROOT = ROOT / "results" / "paper2_final"
GEN_ROOT = OUT_ROOT / "gen"
SUBSET_PATH = OUT_ROOT / "eval_subset.parquet"

DOMAINS = ("athletes", "cities", "writers", "musicians")

# Largest per family + the Polish-tuned mid (PLLuM-12B).
MODELS: list[tuple[str, str]] = [
    ("Bielik-11B", "speakleash/Bielik-11B-v3.0-Instruct"),
    ("gemma-4-12b", "google/gemma-4-12b-it"),
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("PLLuM-12B", "CYFRAGOVPL/PLLuM-12B-instruct-2512"),
]

N_REAL_PER_DECILE = 6
N_FABRICATED = 20
N_SAMPLES = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 64
SUBSET_SEED = 0

# EigenTrack adaptation constants (documented in FINAL_PHASE.md).
EIG_WINDOW = 8          # sliding-window length N (short single-sentence answers)
EIG_N_LAYERS = 8        # ~8 evenly spaced monitored layers (paper: fixed intervals)
EIG_N_FEATURES = 16     # our instantiation of their spectral feature vector
D2H_TOP_K = 0.4         # top-k attention threshold (paper ablation peak: 0.4-0.5)


def slug_for(model_id: str) -> str:
    return model_id.split("/")[-1]


def entity_uid(domain: str, entity: str, qid) -> str:
    """Stable per-entity key. Real entities have a unique Wikidata qid;
    fabricated entities have qid=NaN, so key them by crc32(domain|entity)."""
    if isinstance(qid, str) and qid.strip():
        return qid
    return f"fab-{zlib.crc32(f'{domain}|{entity}'.encode()):08x}"


# ---------------------------------------------------------------------------
# Eval subset construction (pure CPU, deterministic).
# ---------------------------------------------------------------------------
def build_eval_subset() -> "pd.DataFrame":
    import pandas as pd

    rng = np.random.default_rng(SUBSET_SEED)
    frames = []
    for domain in DOMAINS:
        ent = pd.read_parquet(DATA_V2 / f"entities_{domain}.parquet")
        picks = []
        for decile in range(10):
            pool = ent[(ent["kind"] == "real") & (ent["decile"] == decile)]
            idx = rng.choice(pool.index.to_numpy(), size=N_REAL_PER_DECILE, replace=False)
            picks.append(pool.loc[sorted(idx)])
        fab = ent[ent["kind"] == "fabricated"]
        idx = rng.choice(fab.index.to_numpy(), size=N_FABRICATED, replace=False)
        picks.append(fab.loc[sorted(idx)])
        sub = pd.concat(picks, ignore_index=True)
        sub.insert(0, "domain", domain)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "uid", [entity_uid(d, e, q) for d, e, q in
                          zip(out["domain"], out["entity"], out["qid"])])
    if out["uid"].duplicated().any():
        raise ValueError("eval subset uids are not unique")
    return out


# ---------------------------------------------------------------------------
# Spectral features for EigenTrack (pure numpy; unit-testable).
# ---------------------------------------------------------------------------
def mp_reference_eigs(n_rows: int, n_cols: int, sigma2: float, n_grid: int = 64) -> np.ndarray:
    """Quantile grid sample of the Marchenko-Pastur eigenvalue law.

    Reference for the nonzero covariance eigenvalues of a (n_rows x n_cols)
    matrix with iid zero-mean entries of variance sigma2, in the regime
    gamma = n_rows / n_cols (here gamma << 1). Support:
    [sigma2 (1-sqrt(gamma))^2, sigma2 (1+sqrt(gamma))^2].
    """
    gamma = n_rows / n_cols
    lo = sigma2 * (1.0 - np.sqrt(gamma)) ** 2
    hi = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    xs = np.linspace(lo, hi, n_grid * 8)
    with np.errstate(invalid="ignore"):
        dens = np.sqrt(np.maximum((hi - xs) * (xs - lo), 0.0)) / (
            2.0 * np.pi * sigma2 * gamma * np.maximum(xs, 1e-30))
    total = dens.sum()
    if total <= 0:
        return np.full(n_grid, sigma2)
    cdf = np.cumsum(dens) / total
    qs = (np.arange(n_grid) + 0.5) / n_grid
    return np.interp(qs, cdf, xs)


def spectral_features(window: np.ndarray) -> np.ndarray:
    """16 spectral features of a sliding-window activation matrix (n x D).

    Instantiates the EigenTrack feature categories: leading eigenvalues,
    spectral gaps, spectral entropy, spectral variance, central tendency,
    effective rank, and KL/Wasserstein divergence to the Marchenko-Pastur
    reference. Eigenvalues lambda_i = sigma_i^2 / n from the SVD of the
    window (covariance convention of the paper).
    """
    H = np.asarray(window, dtype=np.float64)
    n, d = H.shape
    sv = np.linalg.svd(H, compute_uv=False)
    lam = (sv ** 2) / n
    lam = np.sort(lam)[::-1]
    lam_sum = lam.sum()
    p = lam / lam_sum if lam_sum > 0 else np.full_like(lam, 1.0 / len(lam))
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    top = np.zeros(5)
    top[: min(5, len(lam))] = lam[:5]
    gap12 = float(lam[0] / lam[1]) if len(lam) > 1 and lam[1] > 0 else 0.0
    gap23 = float(lam[1] / lam[2]) if len(lam) > 2 and lam[2] > 0 else 0.0
    sigma2 = float((H ** 2).mean())
    mp = mp_reference_eigs(n, d, max(sigma2, 1e-30))
    # Wasserstein-1 between the empirical eigenvalues and the MP quantile grid
    # (both treated as discrete distributions), scale-normalized by sigma2.
    emp_q = np.interp((np.arange(len(mp)) + 0.5) / len(mp),
                      (np.arange(len(lam)) + 0.5) / len(lam), lam)
    w_mp = float(np.abs(emp_q - mp).mean() / max(sigma2, 1e-30))
    # KL between normalized histograms on shared bins.
    bins = np.linspace(0.0, max(lam.max(), mp.max()) * 1.01 + 1e-30, 17)
    h_emp, _ = np.histogram(lam, bins=bins)
    h_mp, _ = np.histogram(mp, bins=bins)
    pe = (h_emp + 1e-9) / (h_emp + 1e-9).sum()
    pm = (h_mp + 1e-9) / (h_mp + 1e-9).sum()
    kl_mp = float((pe * np.log(pe / pm)).sum())
    feats = np.array([
        *top,                                   # 5: leading eigenvalues
        gap12, gap23,                           # 2: spectral gaps
        entropy,                                # 1: spectral entropy
        float(lam.var()),                       # 1: spectral variance
        float(lam_sum),                         # 1: total spectral power
        float(np.median(lam)),                  # 1: median eigenvalue
        float(np.exp(entropy)),                 # 1: effective rank
        kl_mp, w_mp,                            # 2: divergence to MP baseline
        float(lam[0] / lam_sum) if lam_sum > 0 else 0.0,        # top-1 frac
        float(lam[:3].sum() / lam_sum) if lam_sum > 0 else 0.0,  # top-3 frac
    ], dtype=np.float32)
    assert feats.shape[0] == EIG_N_FEATURES
    return feats


def eigentrack_feature_sequence(vs: np.ndarray, window: int = EIG_WINDOW) -> np.ndarray:
    """Per-step spectral features over a sliding window of token vectors.

    ``vs``: (T, D) concatenated monitored-layer hidden states per response
    token. Steps start at t=2 (>=2 rows needed for a meaningful spectrum);
    the window grows until it reaches ``window`` and then slides.
    Returns (max(T-1, 0), EIG_N_FEATURES).
    """
    T = vs.shape[0]
    seq = []
    for t in range(2, T + 1):
        w = vs[max(0, t - window): t]
        seq.append(spectral_features(w))
    if not seq:
        return np.zeros((0, EIG_N_FEATURES), dtype=np.float32)
    return np.stack(seq)


# ---------------------------------------------------------------------------
# D2HScore raw components (pure numpy over captured tensors; unit-testable).
# ---------------------------------------------------------------------------
def d2h_dispersion_per_layer(hidden_resp: np.ndarray) -> np.ndarray:
    """Intra-layer dispersion D_l. ``hidden_resp``: (L, T, d) response-token
    hidden states for transformer layers 1..L. Returns (L,)."""
    c = hidden_resp.mean(axis=1, keepdims=True)          # (L, 1, d)
    return np.linalg.norm(hidden_resp - c, axis=2).mean(axis=1)


def d2h_drift_per_layer(hidden_resp: np.ndarray, attn_from_last: np.ndarray,
                        top_k: float = D2H_TOP_K) -> np.ndarray:
    """Inter-layer drift delta_l with attention-guided key tokens.

    ``hidden_resp``: (L, T, d); ``attn_from_last``: (L, T) head-averaged
    attention weights from the final response token to each response token,
    per layer. Key set K_l = top ceil(top_k * T) tokens by attention at layer
    l; core repr = mean hidden state over K_l; drift = ||core_{l+1} - core_l||.
    Returns (L-1,).
    """
    L, T, _ = hidden_resp.shape
    k = max(1, int(np.ceil(top_k * T)))
    cores = np.empty((L, hidden_resp.shape[2]))
    for l in range(L):
        key_idx = np.argsort(attn_from_last[l])[::-1][:k]
        cores[l] = hidden_resp[l, key_idx].mean(axis=0)
    return np.linalg.norm(np.diff(cores, axis=0), axis=1)


# ---------------------------------------------------------------------------
# Generation + capture (GPU; one model per process).
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render_prompt(tokenizer, template: str, entity: str, model_id: str) -> str:
    messages = [{"role": "user", "content": template.format(entity=entity)}]
    kwargs = {}
    if "Qwen3" in model_id:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)


def gen_seed(qid: str, sample_idx: int) -> int:
    return zlib.crc32(f"{qid}|{sample_idx}".encode()) & 0x7FFFFFFF


def monitored_layer_indices(n_layers: int, n_monitored: int = EIG_N_LAYERS) -> list[int]:
    """Evenly spaced 1-based transformer-layer indices, always including L."""
    if n_layers <= n_monitored:
        return list(range(1, n_layers + 1))
    idx = np.unique(np.round(np.linspace(1, n_layers, n_monitored)).astype(int))
    return [int(i) for i in idx]


def run_model(model_id: str, limit: int | None = None,
              domains: tuple[str, ...] = DOMAINS) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bielik_hallu.dataset.v2.templates import templates_for

    slug = slug_for(model_id)
    subset = pd.read_parquet(SUBSET_PATH)
    _log(f"loading {model_id} (eager attention for output_attentions)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("mps")
    model.eval()

    special_ids = set(tokenizer.all_special_ids)
    t_model0 = time.time()

    for domain in domains:
        template = templates_for(domain)["prompt_pl"]
        dom_dir = GEN_ROOT / slug / domain
        cap_dir = dom_dir / "cap"
        cap_dir.mkdir(parents=True, exist_ok=True)
        answers_path = dom_dir / "answers.jsonl"
        done: set[str] = set()
        if answers_path.exists():
            with answers_path.open() as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["uid"])

        sub = subset[subset["domain"] == domain]
        if limit is not None:
            sub = sub.head(limit)
        _log(f"{slug}/{domain}: {len(sub)} entities ({len(done)} already done)")
        for _, r in sub.iterrows():
            uid = r["uid"]
            if uid in done and (cap_dir / f"{uid}.npz").exists():
                continue
            t0 = time.time()
            entity = r["entity"]
            prompt = render_prompt(tokenizer, template, entity, model_id)
            enc = tokenizer(prompt, return_tensors="pt").to("mps")
            n_prompt = enc["input_ids"].shape[1]

            answers: list[str] = []
            kept_gen_ids: list[int] | None = None
            for s in range(N_SAMPLES):
                torch.manual_seed(gen_seed(uid, s))
                with torch.no_grad():
                    out = model.generate(
                        **enc, do_sample=True, temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                gen_ids = out[0][n_prompt:].tolist()
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                answers.append(text)
                if s == 0:
                    # Strip leading/trailing special tokens (EOS / end_of_turn /
                    # channel markers); keep interior tokens untouched.
                    while gen_ids and gen_ids[-1] in special_ids:
                        gen_ids.pop()
                    while gen_ids and gen_ids[0] in special_ids:
                        gen_ids.pop(0)
                    kept_gen_ids = gen_ids

            # --- capture forward over prompt + answer#0 ---
            if not kept_gen_ids:  # degenerate: empty sampled answer
                kept_gen_ids = out[0][n_prompt:].tolist()[:1] or [tokenizer.eos_token_id]
            full_ids = torch.tensor(
                [enc["input_ids"][0].tolist() + kept_gen_ids], device="mps")
            attn_mask = torch.ones_like(full_ids)
            with torch.no_grad():
                fwd = model(input_ids=full_ids, attention_mask=attn_mask,
                            output_hidden_states=True, output_attentions=True)
            hs = fwd.hidden_states                    # (L+1) x (1, S, d)
            atts = fwd.attentions                     # L x (1, H, S, S)
            L = len(atts)
            T = len(kept_gen_ids)
            resp = slice(n_prompt, n_prompt + T)
            last_pos = n_prompt + T - 1

            hidden_resp = np.stack([
                hs[l][0, resp].float().cpu().numpy() for l in range(1, L + 1)
            ])                                        # (L, T, d)
            attn_from_last = np.stack([
                atts[l][0, :, last_pos, resp].float().mean(dim=0).cpu().numpy()
                for l in range(L)
            ])                                        # (L, T)

            disp = d2h_dispersion_per_layer(hidden_resp)
            drift = d2h_drift_per_layer(hidden_resp, attn_from_last)

            mon = monitored_layer_indices(L)
            vs = np.concatenate([hidden_resp[l - 1] for l in mon], axis=1)  # (T, m*d)
            eig_seq = eigentrack_feature_sequence(vs)

            mind_last = hidden_resp[L - 1, T - 1].astype(np.float16)
            mind_mean = hidden_resp[L - 1].mean(axis=0).astype(np.float16)

            np.savez_compressed(
                cap_dir / f"{uid}.npz",
                disp=disp.astype(np.float32),
                drift=drift.astype(np.float32),
                eig_seq=eig_seq.astype(np.float32),
                mind_last=mind_last, mind_mean=mind_mean,
                monitored_layers=np.array(mon, dtype=np.int32),
                n_resp_tokens=np.array([T], dtype=np.int32),
            )
            rec = {
                "uid": uid, "qid": r["qid"] if isinstance(r["qid"], str) else None,
                "entity": entity, "domain": domain,
                "kind": r["kind"], "decile": int(r["decile"]),
                "answers": answers, "captured_sample": 0,
                "n_resp_tokens": T, "n_prompt_tokens": int(n_prompt),
                "runtime_s": round(time.time() - t0, 1),
            }
            with answers_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            del fwd, hs, atts, hidden_resp, out
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            _log(f"  {slug}/{domain}/{entity!r}: T={T} "
                 f"({time.time() - t0:.1f}s)")

    meta = {"model_id": model_id, "slug": slug,
            "runtime_s": round(time.time() - t_model0, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (GEN_ROOT / slug / "gen_done.json").write_text(json.dumps(meta, indent=2))
    _log(f"MODEL DONE {slug} in {meta['runtime_s']/3600:.2f} h")


def model_done(model_id: str) -> bool:
    return (GEN_ROOT / slug_for(model_id) / "gen_done.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-subset", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: entities per domain")
    ap.add_argument("--domains", type=str, default=None,
                    help="comma-separated domain filter")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.build_subset or not SUBSET_PATH.exists():
        df = build_eval_subset()
        df.to_parquet(SUBSET_PATH)
        _log(f"wrote {SUBSET_PATH}: {len(df)} rows "
             f"({df.groupby(['domain', 'kind']).size().to_dict()})")
        if args.build_subset and not (args.model or args.all):
            return

    if args.model:
        doms = tuple(args.domains.split(",")) if args.domains else DOMAINS
        run_model(args.model, limit=args.limit, domains=doms)
    elif args.all:
        for label, model_id in MODELS:
            if model_done(model_id):
                _log(f"SKIP {label} (gen_done.json present)")
                continue
            _log(f"=== spawning child for {label} ({model_id}) ===")
            ret = subprocess.run(
                [sys.executable, __file__, "--model", model_id],
                cwd=str(ROOT)).returncode
            if ret != 0:
                _log(f"CHILD FAILED for {label} (exit {ret}); continuing")


if __name__ == "__main__":
    main()
