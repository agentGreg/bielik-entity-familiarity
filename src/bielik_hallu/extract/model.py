import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bielik_hallu import config


def decoder_module(model):
    """Return the text decoder submodule that owns ``.layers`` and ``.norm``.

    Bielik (LlamaForCausalLM-style) exposes it at ``model.model``. Gemma-3
    instruct checkpoints load via AutoModelForCausalLM as a multimodal
    ``Gemma3ForConditionalGeneration`` whose LM lives under
    ``model.model.language_model``. Detect the language_model wrapper and fall
    back to ``model.model`` so the same hook/extraction code serves both.
    """
    base = model.model
    return base.language_model if hasattr(base, "language_model") else base


def final_norm(model):
    """Return the final RMSNorm applied before ``lm_head`` (for logit-lens)."""
    return decoder_module(model).norm


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=getattr(torch, config.DTYPE),
        output_hidden_states=True,
    ).to(config.DEVICE)
    model.eval()
    return model, tokenizer


class ActivationCapturer:
    """Context manager: captures each MLP layer's act_fn output."""

    def __init__(self, model):
        self.model = model
        self.activations: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inp, out):
            self.activations[layer_idx] = out.detach()[0].float().cpu()
        return hook

    def __enter__(self):
        for i, layer in enumerate(decoder_module(self.model).layers):
            h = layer.mlp.act_fn.register_forward_hook(self._make_hook(i))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.activations.clear()
