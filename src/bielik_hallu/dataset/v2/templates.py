"""Prompt templates for dataset v2 (paper 2, RQ2).

Two language variants (PL / EN) per domain type, plus a neutral stem pair used
downstream for open-ended probing. People domains (athletes, writers, musicians)
share the "who is" phrasing; the cities/localities domain uses "what is".

Entity display titles are domain data and stay in their native form; the template
strings themselves are the only language-carrying constants here.

Mirrors the v1 convention in ``candidates_domains.py`` (``PROMPT_TEMPLATE_PEOPLE``
/ ``PROMPT_TEMPLATE_PLACES``) so v1<->v2 prompts are comparable — the PL people /
places strings are byte-identical to v1.
"""

from __future__ import annotations

# --- v1-identical primary templates (comparability) ------------------------
PROMPT_TEMPLATE_PEOPLE_PL = "Kim jest {entity}? Odpowiedz jednym zdaniem."
PROMPT_TEMPLATE_PLACES_PL = "Czym jest {entity}? Odpowiedz jednym zdaniem."

# --- English counterparts --------------------------------------------------
PROMPT_TEMPLATE_PEOPLE_EN = "Who is {entity}? Answer in one sentence."
PROMPT_TEMPLATE_PLACES_EN = "What is {entity}? Answer in one sentence."

# --- neutral open-ended stem pair (constants for later use) -----------------
NEUTRAL_STEM_PL = "Co wiesz o {entity}? Odpowiedz jednym zdaniem."
NEUTRAL_STEM_EN = "What do you know about {entity}? Answer in one sentence."

# Which template family a domain uses.
_PEOPLE_DOMAINS = {"athletes", "writers", "musicians"}
_PLACE_DOMAINS = {"cities"}


def templates_for(domain: str) -> dict[str, str]:
    """Return the PL/EN primary templates plus neutral stems for a domain."""
    if domain in _PEOPLE_DOMAINS:
        pl, en = PROMPT_TEMPLATE_PEOPLE_PL, PROMPT_TEMPLATE_PEOPLE_EN
    elif domain in _PLACE_DOMAINS:
        pl, en = PROMPT_TEMPLATE_PLACES_PL, PROMPT_TEMPLATE_PLACES_EN
    else:  # pragma: no cover - guard against typos in callers
        raise KeyError(f"unknown domain: {domain!r}")
    return {
        "prompt_pl": pl,
        "prompt_en": en,
        "neutral_pl": NEUTRAL_STEM_PL,
        "neutral_en": NEUTRAL_STEM_EN,
    }


# Public per-domain mapping, mirroring the v1 DOMAINS[...] shape.
DOMAINS = ("athletes", "cities", "writers", "musicians")
TEMPLATES = {d: templates_for(d) for d in DOMAINS}
