from pathlib import Path
import numpy as np
import pandas as pd
import torch

from bielik_hallu import config
from bielik_hallu.metrics.dispersion import ipr, shannon_entropy, normalize_outliers
from bielik_hallu.metrics.baseline import first_token_logprob_entropy
from bielik_hallu.extract.model import load_model, ActivationCapturer
from bielik_hallu.extract.positions import find_entity_last_token_in_offsets
from bielik_hallu.dataset.label import render_prompt


def metrics_for_activation(vec: np.ndarray) -> dict:
    v = normalize_outliers(np.asarray(vec, dtype=np.float64))
    return {"ipr": ipr(v), "entropy": shannon_entropy(v)}


def extract_signals(labeled_path: Path, results_dir: Path | None = None,
                    template: str | None = None) -> Path:
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    model, tokenizer = load_model()
    df = pd.read_parquet(labeled_path)
    rows = []
    hidden_entity_by_layer: dict[int, list[np.ndarray]] = {}
    hidden_prompt_by_layer: dict[int, list[np.ndarray]] = {}

    for _, r in df.iterrows():
        rendered = render_prompt(tokenizer, r["entity"], template)
        # Tokenize ONCE with offsets and feed those exact ids to the model, so ent_idx
        # is computed against the SAME encoding (including any prepended special
        # tokens like BOS) that the model actually consumes.
        enc = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        ent_idx = find_entity_last_token_in_offsets(offsets, rendered, r["entity"])
        prm_idx = input_ids.shape[1] - 1

        with ActivationCapturer(model) as cap, torch.no_grad():
            # output_hidden_states passed explicitly at call time: the Gemma-3
            # multimodal wrapper ignores the config flag set at load time, and
            # passing it here is harmless for Bielik.
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        first_logits = out.logits[0, -1].float().cpu().numpy()
        ft_entropy = first_token_logprob_entropy(first_logits)
        hidden_states = out.hidden_states  # tuple: (n_layers+1) x (1, seq, hidden)

        for layer, act in cap.activations.items():
            for point, idx in (("entity", ent_idx), ("prompt", prm_idx)):
                vec = act[idx].numpy()
                m = metrics_for_activation(vec)
                rows.append({
                    "entity": r["entity"], "condition": r["condition"],
                    "label_hallucination": int(r["label_hallucination"]),
                    "layer": layer, "point": point,
                    "ipr": m["ipr"], "entropy": m["entropy"],
                    "first_token_entropy": ft_entropy,
                })
        # residual-stream hidden states for the probe, at BOTH measurement points, all layers
        for layer in range(len(hidden_states)):
            h_entity = hidden_states[layer][0, ent_idx].float().cpu().numpy()
            h_prompt = hidden_states[layer][0, prm_idx].float().cpu().numpy()
            hidden_entity_by_layer.setdefault(layer, []).append(h_entity)
            hidden_prompt_by_layer.setdefault(layer, []).append(h_prompt)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "signals.parquet"
    pd.DataFrame(rows).to_parquet(out_path)
    np.savez(results_dir / "hidden_states.npz",
             labels=df["label_hallucination"].to_numpy(),
             # Save as fixed-width unicode (not object dtype) so the npz can be
             # loaded with allow_pickle=False (pandas .to_numpy() gives object).
             conditions=df["condition"].to_numpy(dtype="U"),
             **{f"entity_layer_{k}": np.stack(v) for k, v in hidden_entity_by_layer.items()},
             **{f"prompt_layer_{k}": np.stack(v) for k, v in hidden_prompt_by_layer.items()})
    return out_path
