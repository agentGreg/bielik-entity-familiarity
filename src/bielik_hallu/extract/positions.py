def last_prompt_token_index(input_ids) -> int:
    return len(input_ids) - 1


def find_entity_last_token_in_offsets(offset_mapping, rendered: str, entity: str) -> int:
    """Return the last token index overlapping the entity span, given a GIVEN offset_mapping.

    This is pure and works on any offset_mapping (list of (start, end) tuples), so callers
    can pass the offsets from the exact encoding fed to the model (which may include
    special tokens like BOS at (0, 0)) and get an index that aligns with that sequence.
    """
    ent_start = rendered.index(entity)
    ent_end = ent_start + len(entity)
    last = None
    for i, (s, e) in enumerate(offset_mapping):
        if e <= s:  # empty / special tokens
            continue
        # A token belongs to the entity if it ends within the entity span
        # (e <= ent_end) and covers at least one entity character (e > ent_start).
        # The start may fall a few characters before ent_start when the tokenizer
        # merges the preceding whitespace into the entity's first subword — a
        # SentencePiece/BPE leading-space token, e.g. offsets (29, 38) for
        # " Warszawa" when the entity "Warszawa" starts at 30. Such a token is
        # still the entity's token, so accept it as long as everything before
        # ent_start in its span is whitespace (never letters of another word).
        if e <= ent_end and e > ent_start and rendered[s:ent_start].strip() == "":
            last = i
    if last is None:
        raise ValueError(f"No entity tokens found for {entity!r} in the prompt")
    return last


def find_entity_last_token(tokenizer, rendered: str, entity: str) -> int:
    """Return the last token index overlapping the entity span in the rendered prompt."""
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    return find_entity_last_token_in_offsets(enc["offset_mapping"], rendered, entity)
