from bielik_hallu import config


def render_prompt(tokenizer, entity: str, template: str | None = None) -> str:
    """Render the chat-templated prompt for an entity.

    ``template`` defaults to ``config.PROMPT_TEMPLATE`` (the athletes/people
    "Kim jest {entity}?" prompt). Pass an explicit template for other domains,
    e.g. the cities "Czym jest {entity}?" prompt used by the E6 extension.
    """
    if template is None:
        template = config.PROMPT_TEMPLATE
    messages = [{"role": "user", "content": template.format(entity=entity)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_answers(model, tokenizer, entity: str, n: int | None = None,
                   temperature: float = 0.7) -> list[str]:
    n = config.N_SAMPLES if n is None else n
    prompt = render_prompt(tokenizer, entity)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    answers = []
    for _ in range(n):
        out = model.generate(**inputs, do_sample=True, temperature=temperature,
                             max_new_tokens=64)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        answers.append(text.strip())
    return answers


def judge_known(entity: str, answers: list[str], judge_fn) -> bool:
    """KNOWN is correct only if all samples are judged correct."""
    return all(judge_fn(entity, a) for a in answers)


def label_row(condition: str, all_correct: bool) -> int:
    if condition == "KNOWN":
        return 0 if all_correct else 1
    return 1  # UNKNOWN_REAL, FABRICATED -> hallucination by default
