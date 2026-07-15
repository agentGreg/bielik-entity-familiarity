"""Fabricated-anchor construction for dataset v2 (paper 2, RQ2).

Mirrors the v1 methodology (``candidates_domains.py``): morphologically valid,
invented Polish names/toponyms, **token-count matched** to the REAL sample under
BOTH tokenizer families (Bielik-1.5B and Bielik-11B, which tokenize very
differently), web-checked for non-existence against pl.wiki. We seed the
candidate pool from v1's fabricated lists (flag ``in_v1``) and generate the rest
by light morphological mutation of real Polish name/toponym morphology.

Selection objective (token-length matching): from a screened, non-existent candidate
pool, pick the subset whose per-name token counts make the real-vs-fabricated
K-token AUROC closest to 0.5 under *both* tokenizers (target < 0.65 each).

Tokenizers are loaded from the local HF cache (offline). No GPU needed — the
tokenizer is a pure CPU BPE/SentencePiece model.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache

import numpy as np

# Keep tokenizer loading fully offline (HF cache only).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The two tokenizer families to match under (both v3.0 Instruct).
TOKENIZERS = {
    "bielik_1.5b": "speakleash/Bielik-1.5B-v3.0-Instruct",
    "bielik_11b": "speakleash/Bielik-11B-v3.0-Instruct",
}


@lru_cache(maxsize=None)
def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def token_count(text: str, family: str) -> int:
    tok = _tokenizer(TOKENIZERS[family])
    return len(tok(text, add_special_tokens=False)["input_ids"])


def token_counts(text: str) -> dict[str, int]:
    return {fam: token_count(text, fam) for fam in TOKENIZERS}


# --------------------------------------------------------------------------
# K-token AUROC (real vs fabricated) from token counts alone.
# --------------------------------------------------------------------------
def ktoken_auroc(real_counts: list[int], fab_counts: list[int]) -> float:
    """AUROC of a classifier using only token count to tell real from
    fabricated. 0.5 = token length carries no signal. Symmetric: we report the
    max of (score, 1-score) so direction does not matter (we want it near 0.5).
    """
    if not real_counts or not fab_counts:
        return float("nan")
    # Mann-Whitney U -> AUROC. P(fab_count > real_count) + 0.5 ties.
    wins = 0.0
    for f in fab_counts:
        for r in real_counts:
            if f > r:
                wins += 1.0
            elif f == r:
                wins += 0.5
    auc = wins / (len(real_counts) * len(fab_counts))
    return max(auc, 1.0 - auc)


# --------------------------------------------------------------------------
# Morphological generation of invented Polish names / toponyms.
# --------------------------------------------------------------------------
# Productive Polish surname stems + suffixes (people). Light, athletes-list
# style: recombine a stem with a surname suffix. Not meant to be exhaustive,
# only to produce a large screened pool for token-matched selection.
_SURNAME_STEMS = [
    "Kowal", "Nowak", "Wiśniew", "Wójcik", "Kowalczyk", "Kamiń", "Lewandow",
    "Zieliń", "Szymań", "Woźniak", "Dąbrow", "Koz", "Jankow", "Mazur",
    "Krawczyk", "Piotrow", "Grabow", "Nowicki", "Pawłow", "Michalak",
    "Król", "Wieczor", "Jabłoń", "Wróbel", "Stępień", "Górski", "Rutkow",
    "Michal", "Sikor", "Baran", "Duda", "Szewczyk", "Tomasz", "Pietrzak",
    "Wróblew", "Marcin", "Zając", "Pawlak", "Witkow", "Walczak", "Sokołow",
    "Urbań", "Rybak", "Głowacki", "Malec", "Adam", "Sobczak", "Czarniecki",
]
_SURNAME_SUFFIXES = [
    "ski", "cki", "wicz", "czyk", "ak", "ek", "iński", "owicz", "owski",
    "arczyk", "owiak", "ewski", "icki", "ecki",
]
_FIRST_NAMES_M = [
    "Adam", "Andrzej", "Antoni", "Bartosz", "Bogdan", "Cezary", "Dawid",
    "Damian", "Filip", "Grzegorz", "Henryk", "Ignacy", "Jakub", "Jan",
    "Jarosław", "Konrad", "Leszek", "Marek", "Mateusz", "Michał", "Mikołaj",
    "Patryk", "Paweł", "Piotr", "Rafał", "Remigiusz", "Stanisław", "Szymon",
    "Tadeusz", "Tomasz", "Wojciech", "Zbigniew",
]
_FIRST_NAMES_F = [
    "Agata", "Agnieszka", "Alicja", "Anna", "Barbara", "Beata", "Danuta",
    "Dorota", "Edyta", "Ewa", "Halina", "Irena", "Iwona", "Joanna", "Justyna",
    "Karolina", "Katarzyna", "Magdalena", "Małgorzata", "Marta", "Natalia",
    "Roksana", "Sylwia", "Wanda", "Zofia",
]

# Toponym stems + suffixes (places).
_TOPO_STEMS = [
    "Brzoz", "Brzez", "Cis", "Cyran", "Dębo", "Grab", "Jawor", "Kalin",
    "Kawk", "Klon", "Krzew", "Kun", "Lip", "Modrzew", "Młyn", "Olch",
    "Plisz", "Sarn", "Sikor", "Szczygl", "Topol", "Wierzb", "Wilg", "Wydrz",
    "Sosn", "Buk", "Świerk", "Jodł", "Leszcz", "Kalisz", "Bagn", "Kamien",
]
_TOPO_SUFFIXES = [
    "ice", "ów", "owo", "iny", "niki", "sko", "ęcin", "no", "owice",
    "owiec", "ówka", "owa", "any", "nica",
]
# Multi-word toponym qualifiers — real Polish village names are frequently
# compound ("Szeligi Górne", "Nowe Przyłuski", "Wola Grabowska", "Pijanów-
# Kolonia"). The v2 REAL locality sample skews long/multi-word (median ~5 tok
# under the 1.5B tokenizer, ~60% contain a space), so a single-word fabricated
# pool is token-separable. These qualifiers lengthen fabricated toponyms to
# match that distribution.
_TOPO_ADJ = [
    "Górne", "Dolne", "Wielkie", "Małe", "Nowe", "Stare", "Wyżne", "Niżne",
    "Górna", "Dolna", "Wielka", "Mała", "Nowa", "Stara", "Wyżna", "Niżna",
    "Górny", "Dolny", "Wielki", "Mały", "Nowy", "Stary",
]
_TOPO_PREFIX = ["Wola", "Nowa", "Stara", "Nowe", "Stare"]
_TOPO_TAIL = ["Kolonia", "Wieś", "Poduchowna"]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# Short surname roots (mutated to invented forms via a short suffix or a single
# consonant/vowel swap) — these keep some fabricated names near the token length
# of short famous names, widening the pool for token-matched selection.
_SHORT_ROOTS = [
    "Bąk", "Bór", "Cieśl", "Dąb", "Fik", "Gil", "Gaj", "Hab", "Jaś", "Kęs",
    "Kos", "Kot", "Lis", "Łoś", "Miś", "Puc", "Rak", "Sęp", "Sów", "Wid",
    "Wilk", "Zub", "Żak", "Bal", "Cyl", "Dyl", "Fal", "Gruz", "Kula", "Rok",
]
_SHORT_SUFFIXES = ["ik", "ek", "ka", "ko", "el", "yk", "a", "o"]


def _generate_people(rng: random.Random, n: int) -> list[str]:
    out: set[str] = set()
    firsts = _FIRST_NAMES_M + _FIRST_NAMES_F
    guard = 0
    while len(out) < n and guard < n * 400:
        guard += 1
        first = rng.choice(firsts)
        # ~40% short surnames for token-length diversity, else compound.
        if rng.random() < 0.4:
            surname = _cap(rng.choice(_SHORT_ROOTS)) + rng.choice(_SHORT_SUFFIXES)
        else:
            surname = _cap(rng.choice(_SURNAME_STEMS)) + rng.choice(_SURNAME_SUFFIXES)
        out.add(f"{first} {surname}")
    return list(out)


def _generate_places(rng: random.Random, n: int) -> list[str]:
    out: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 400:
        guard += 1
        base = _cap(rng.choice(_TOPO_STEMS)) + rng.choice(_TOPO_SUFFIXES)
        # Mix single-word and multi-word forms to span the real length range.
        roll = rng.random()
        if roll < 0.35:                       # single word (short tail)
            name = base
        elif roll < 0.70:                     # base + directional adjective
            name = f"{base} {rng.choice(_TOPO_ADJ)}"
        elif roll < 0.85:                     # prefix + base
            name = f"{rng.choice(_TOPO_PREFIX)} {base}"
        elif roll < 0.93:                     # hyphenated compound tail
            name = f"{base}-{rng.choice(_TOPO_TAIL)}"
        else:                                 # two stems compounded
            name = f"{base} {_cap(rng.choice(_TOPO_STEMS))}{rng.choice(_TOPO_SUFFIXES)}"
        out.add(name)
    return list(out)


def generate_candidates(domain: str, seed_names: list[str], rng: random.Random,
                        pool_size: int = 400) -> list[dict]:
    """Build a fabricated-name candidate pool for a domain.

    Seeds first (v1 fabricated names, flagged in_v1=True), then programmatic
    morphological generation to fill ``pool_size``. Returns dicts with keys
    ``name`` and ``in_v1``. Non-existence is NOT yet checked here.
    """
    is_place = domain == "cities"
    pool: dict[str, bool] = {}  # name -> in_v1
    for nm in seed_names:
        pool[nm] = True
    need = max(0, pool_size - len(pool))
    gen = (_generate_places(rng, need * 2) if is_place
           else _generate_people(rng, need * 2))
    rng.shuffle(gen)
    for nm in gen:
        if nm not in pool:
            pool[nm] = False
        if len(pool) >= pool_size:
            break
    return [{"name": nm, "in_v1": iv} for nm, iv in pool.items()]


# --------------------------------------------------------------------------
# Token-matched selection.
# --------------------------------------------------------------------------
def select_token_matched(
    real_names: list[str],
    fab_candidates: list[dict],
    n_select: int,
    rng: random.Random,
    n_restarts: int = 400,
    target_auroc: float = 0.65,
) -> tuple[list[dict], dict]:
    """Select ``n_select`` fabricated names whose token counts minimise the
    worse-of-two-tokenizers K-token AUROC vs the real sample.

    Greedy random-restart: try many random subsets, keep the one with the
    lowest max(AUROC_1.5b, AUROC_11b). Deterministic given ``rng``.

    Returns (selected candidate dicts with per-family token counts attached,
    diagnostics dict with the achieved AUROCs).
    """
    real_1 = [token_count(n, "bielik_1.5b") for n in real_names]
    real_11 = [token_count(n, "bielik_11b") for n in real_names]

    enriched = []
    for c in fab_candidates:
        enriched.append({
            **c,
            "tok_1.5b": token_count(c["name"], "bielik_1.5b"),
            "tok_11b": token_count(c["name"], "bielik_11b"),
        })
    if len(enriched) < n_select:
        raise ValueError(
            f"only {len(enriched)} candidates, need {n_select}")

    idx = list(range(len(enriched)))
    best = None
    best_score = float("inf")
    best_aucs = None
    # Always seed one restart that maximally prefers in_v1 names.
    v1_first = sorted(idx, key=lambda i: (not enriched[i]["in_v1"],
                                          rng.random()))
    restarts = [v1_first[:n_select]]
    for _ in range(n_restarts):
        restarts.append(rng.sample(idx, n_select))
    for sel in restarts:
        f1 = [enriched[i]["tok_1.5b"] for i in sel]
        f11 = [enriched[i]["tok_11b"] for i in sel]
        a1 = ktoken_auroc(real_1, f1)
        a11 = ktoken_auroc(real_11, f11)
        score = max(a1, a11)
        # tie-break: prefer more in_v1 seeds retained
        v1kept = sum(enriched[i]["in_v1"] for i in sel)
        score_adj = score - 1e-4 * v1kept
        if score_adj < best_score:
            best_score = score_adj
            best = sel
            best_aucs = (a1, a11)
    selected = [enriched[i] for i in best]
    diag = {
        "auroc_bielik_1.5b": round(best_aucs[0], 4),
        "auroc_bielik_11b": round(best_aucs[1], 4),
        "target_auroc": target_auroc,
        "passes_target": bool(max(best_aucs) < target_auroc),
        "n_in_v1_selected": int(sum(c["in_v1"] for c in selected)),
        "real_median_tok_1.5b": float(np.median(real_1)),
        "real_median_tok_11b": float(np.median(real_11)),
        "fab_median_tok_1.5b": float(np.median([c["tok_1.5b"] for c in selected])),
        "fab_median_tok_11b": float(np.median([c["tok_11b"] for c in selected])),
    }
    return selected, diag
