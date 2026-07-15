"""Shared inference logic for the hallucination-risk tool.

This module holds the reusable pieces of the risk tool so both the CLI
(``scripts/hallucination_risk.py``) and the web demo (``app.py``) can import them
instead of duplicating:

  - prompt normalization to the calibrated template,
  - chat-template rendering,
  - device / dtype selection (cuda > mps > cpu; float32 on CPU),
  - model + tokenizer loading,
  - one forward pass for the risk score + answer generation,
  - the honest bilingual disclaimer.

The probe NEVER sees the generated answer: it reads prompt-side activations only,
so the risk estimate is invariant to generation settings.
"""
from __future__ import annotations

from bielik_hallu.risk.probe import RiskProbe, risk_band

# The probe is calibrated on prompts ending with this Polish instruction suffix
# (the exact training template was "Kim/Czym jest {entity}? Odpowiedz jednym
# zdaniem."). Bare questions without the suffix are OFF-DISTRIBUTION and can score
# spuriously; we append the suffix to keep real usage on the calibrated manifold.
TEMPLATE_SUFFIX = "Odpowiedz jednym zdaniem."

DISCLAIMER = (
    "Note: this estimate measures the model's familiarity with the entity/topic "
    "from its internal activations. It is calibrated on one-sentence Polish entity "
    'questions ("Kim jest X?" / "Czym jest X?"). It does NOT fact-check the answer '
    "when the model does know the topic, and it is experimental for other question "
    "shapes.\n"
    "Uwaga: to oszacowanie mierzy znajomość encji/tematu na podstawie wewnętrznych "
    'aktywacji modelu. Jest kalibrowane na jednozdaniowych pytaniach ("Kim jest X?" '
    '/ "Czym jest X?"). NIE weryfikuje faktów, gdy model temat zna, i jest '
    "eksperymentalne dla innych typów pytań."
)


def normalize_question(question: str) -> str:
    """Append the calibration template suffix unless the user already included it.

    The probe was trained on "... Odpowiedz jednym zdaniem." prompts; matching that
    at inference keeps the risk estimate on the calibrated distribution. If the user
    already ended with an "answer in one sentence" instruction (either language), we
    leave the question untouched.
    """
    q = question.strip()
    low = q.lower()
    if "jednym zdaniem" in low or "one sentence" in low:
        return q
    sep = " " if q.endswith(("?", ".", "!")) else "? "
    return f"{q}{sep}{TEMPLATE_SUFFIX}"


def render_prompt(tokenizer, question: str) -> str:
    """Chat-template the (normalized) user question ('Kim/Czym jest ...?')."""
    messages = [{"role": "user", "content": normalize_question(question)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def select_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str):
    """Pick a compute dtype for the device.

    On accelerators (cuda / mps) we use bf16 (the model's native dtype). On CPU
    bf16 matmul is slow/poorly supported, so we load in float32 for correctness
    and predictable latency at the cost of ~2x memory.
    """
    import torch

    return torch.float32 if device == "cpu" else torch.bfloat16


def load_runtime(model_id: str, *, device: str | None = None, token: str | None = None):
    """Load model + tokenizer with auto device/dtype selection.

    Mirrors ``bielik_hallu.extract.model.load_model`` but device-agnostic so it
    runs on CPU (Hugging Face Spaces) as well as MPS (local) and CUDA. ``token``
    is forwarded to ``from_pretrained`` for the gated Bielik license (read from a
    Space secret / env by the caller); ``None`` falls back to the local HF cache
    credentials.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = select_device()
    dtype = select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        output_hidden_states=True,
        token=token,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def score_and_answer(
    model,
    tokenizer,
    device,
    probe: RiskProbe,
    question: str,
    *,
    sample: bool = False,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
) -> dict:
    """One forward pass for the risk score (prompt-side), then generate the answer."""
    import torch

    rendered = render_prompt(tokenizer, question)
    enc = tokenizer(rendered, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prm_idx = input_ids.shape[1] - 1  # last prompt token (prompt point)

    # --- risk: prompt-side hidden state at the probe's layer, ONE forward pass ---
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    # hidden_states: tuple (n_layers+1) x (1, seq, hidden); residual axis 0..N.
    h = out.hidden_states[probe.layer][0, prm_idx].float().cpu().numpy()
    raw_score = float(probe.raw_score(h))
    p_risk = float(probe.predict_proba(h))
    band = risk_band(p_risk)

    # --- answer: continue from the SAME prompt (probe never sees this) ---
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )
    answer = tokenizer.decode(
        gen[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    return {
        "question": question,
        "answer": answer,
        "p_risk": p_risk,
        "band": band,
        "raw_score": raw_score,
        "layer": probe.layer,
        "model": probe.metadata.get("model_slug", ""),
    }
