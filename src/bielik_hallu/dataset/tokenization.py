def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenization_metadata(tokenizer, entity: str) -> dict:
    return {"n_tokens_entity": token_length(tokenizer, entity)}
