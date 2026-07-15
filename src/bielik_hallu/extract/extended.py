"""Extended activation metrics (E2): effective rank, attention entropy, logit-lens entropy.

Layer-index caveat (same convention as run_mvp.py): the metric families live in
different native index spaces that happen to share the "layer" column here.

- ``effective_rank`` and ``logitlens_entropy`` are residual-stream metrics,
  layers 0..N (layer 0 is the embedding output).
- ``attn_entropy_mean`` / ``attn_entropy_norm_mean`` are attention-block
  metrics, layers 0..N-1; the layer-N rows carry NaN for them.

Values at the same "layer" value therefore do NOT refer to the same module
across metric families; analyze each metric on its own native axis.

``effective_rank`` is a per-layer quantity computed over ALL prompt tokens
(seq_len x d_hidden), not a per-point quantity. Documented choice: the same
per-layer value is stored on BOTH the 'entity' and 'prompt' rows of a given
layer, so per-(layer, point) groupbys work uniformly across metrics (the
prompt/entity AUROC curves for effective_rank are consequently identical).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bielik_hallu import config
from bielik_hallu.dataset.label import render_prompt
from bielik_hallu.extract.model import final_norm
from bielik_hallu.extract.positions import find_entity_last_token_in_offsets

EXTENDED_METRICS = ("effective_rank", "attn_entropy_mean",
                    "attn_entropy_norm_mean", "logitlens_entropy")

# Column schema for the per-head attention entropy output (E7). One row per
# (entity, layer, head, point); attn_entropy is the raw Shannon entropy over
# causally-visible key positions and attn_entropy_norm divides by log(n_keys).
ATTN_PER_HEAD_COLUMNS = ("entity", "condition", "layer", "head", "point",
                         "attn_entropy", "attn_entropy_norm")


def effective_rank(hidden_matrix: np.ndarray) -> float:
    """Effective rank of a (seq_len x d_hidden) matrix.

    exp of the Shannon entropy of the normalized squared singular values:
    p_i = s_i^2 / sum_j s_j^2, ER = exp(-sum_i p_i log p_i).
    SVD is done in float64 on CPU (numpy); never run this on MPS tensors.
    """
    m = np.asarray(hidden_matrix, dtype=np.float64)
    s = np.linalg.svd(m, compute_uv=False)
    sq = s ** 2
    total = sq.sum()
    if total == 0:
        return float("nan")  # degenerate all-zero matrix
    p = sq / total
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def attention_entropy(attn_probs: np.ndarray) -> tuple[float, float]:
    """Shannon entropy of one attention distribution over key positions.

    Returns (raw, normalized): normalized divides by log(n_keys), the maximum
    achievable entropy over that many valid (causally visible) key positions.
    With a single key position the distribution is forced, so normalized is 0.
    The input is renormalized to sum to 1 to absorb low-precision drift.
    """
    p = np.asarray(attn_probs, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return float("nan"), float("nan")
    p = p / total
    nz = p[p > 0]
    raw = float(-(nz * np.log(nz)).sum())
    if p.size < 2:
        return raw, 0.0
    return raw, float(raw / np.log(p.size))


def attention_entropy_per_head(attn_layer: np.ndarray, query_idx: int) -> list[tuple[float, float]]:
    """Per-head attention entropy for one layer at one query (measurement) point.

    ``attn_layer`` is a (heads, seq, seq) attention-probability tensor for one
    layer (as returned by ``out.attentions[layer][0]``). For the given
    ``query_idx`` only the causally visible key positions 0..query_idx are used
    (autoregressive mask). Returns one ``(raw, normalized)`` entropy tuple per
    head, in head order — the same values whose per-head means become
    ``attn_entropy_mean`` / ``attn_entropy_norm_mean``.
    """
    return [attention_entropy(attn_layer[h, query_idx, :query_idx + 1])
            for h in range(attn_layer.shape[0])]


def logit_lens_entropy(logits: np.ndarray) -> float:
    """Shannon entropy of softmax(logits) over the vocabulary.

    Computed in float64 with max-subtraction for numerical stability; feed
    logits as a CPU numpy array (never compute the softmax on MPS).
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def load_model_eager():
    """Load the model with eager attention so attention weights are returned.

    Deliberately separate from extract.model.load_model: the MVP pipeline
    keeps its default (faster) attention implementation, which cannot return
    per-head attention matrices.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=getattr(torch, config.DTYPE),
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
    ).to(config.DEVICE)
    model.eval()
    return model, tokenizer


def extract_extended_signals(labeled_path: Path, results_dir: Path | None = None,
                             template: str | None = None) -> Path:
    """One forward pass per prompt; write extended_signals.parquet.

    Columns: entity, condition, layer, point, attn_entropy_mean,
    attn_entropy_norm_mean, logitlens_entropy, effective_rank.
    See the module docstring for the layer-index and effective_rank caveats.

    Also writes ``attn_per_head.parquet`` (E7) in the same directory, with one
    row per (entity, condition, layer, head, point): the per-head attention
    entropy at both measurement points. The per-layer ``attn_entropy_mean`` /
    ``attn_entropy_norm_mean`` columns are the mean over heads of exactly those
    per-head values, so the two files stay consistent by construction.

    ``results_dir`` defaults to ``config.RESULTS_DIR`` so existing callers keep
    the per-model layout; pass an explicit directory (e.g. the per-domain
    ``results/<slug>/domains/<domain>/``) to redirect both output files.
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    model, tokenizer = load_model_eager()
    df = pd.read_parquet(labeled_path)
    rows = []
    per_head_rows = []

    for _, r in df.iterrows():
        rendered = render_prompt(tokenizer, r["entity"], template)
        # Tokenize ONCE with offsets and feed those exact ids to the model
        # (same alignment guarantee as extract.run.extract_signals).
        enc = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        ent_idx = find_entity_last_token_in_offsets(offsets, rendered, r["entity"])
        prm_idx = input_ids.shape[1] - 1
        points = (("entity", ent_idx), ("prompt", prm_idx))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=True, output_hidden_states=True)

        n_resid = len(out.hidden_states)  # N+1 residual-stream layers
        n_attn = len(out.attentions)      # N attention blocks

        # Logit lens: batch both measurement points across all residual layers
        # through the final norm + lm_head in one matmul on the device; the
        # entropy itself is then computed on CPU in float64.
        stacked = torch.cat(
            [out.hidden_states[k][0, [ent_idx, prm_idx]] for k in range(n_resid)], dim=0)
        with torch.no_grad():
            lens_logits = model.lm_head(final_norm(model)(stacked)).float().cpu().numpy()

        for layer in range(n_resid):
            # Effective rank over ALL prompt tokens of this residual layer.
            hidden = out.hidden_states[layer][0].float().cpu().numpy()
            er = effective_rank(hidden)

            attn = (out.attentions[layer][0].float().cpu().numpy()
                    if layer < n_attn else None)  # (heads, seq, seq)

            for j, (point, idx) in enumerate(points):
                if attn is not None:
                    per_head = attention_entropy_per_head(attn, idx)
                    attn_mean = float(np.mean([raw for raw, _ in per_head]))
                    attn_norm_mean = float(np.mean([nrm for _, nrm in per_head]))
                    # E7: persist each head's entropy at this (layer, point).
                    for head, (raw, nrm) in enumerate(per_head):
                        per_head_rows.append({
                            "entity": r["entity"], "condition": r["condition"],
                            "layer": layer, "head": head, "point": point,
                            "attn_entropy": raw, "attn_entropy_norm": nrm,
                        })
                else:
                    attn_mean = attn_norm_mean = float("nan")

                rows.append({
                    "entity": r["entity"], "condition": r["condition"],
                    "layer": layer, "point": point,
                    "attn_entropy_mean": attn_mean,
                    "attn_entropy_norm_mean": attn_norm_mean,
                    "logitlens_entropy": logit_lens_entropy(lens_logits[2 * layer + j]),
                    "effective_rank": er,
                })

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "extended_signals.parquet"
    pd.DataFrame(rows).to_parquet(out_path)
    # Enforce a stable column order even when no attention rows exist (n_attn=0).
    per_head_df = pd.DataFrame(per_head_rows, columns=list(ATTN_PER_HEAD_COLUMNS))
    per_head_df.to_parquet(results_dir / "attn_per_head.parquet")
    return out_path
