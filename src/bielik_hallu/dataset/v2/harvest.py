"""Wikidata / Wikipedia harvest client for dataset v2 (paper 2, RQ2).

Reuses the polite API-client patterns from ``scripts/p0_popularity_pilot.py``
(retry-with-backoff, small sleeps, research User-Agent) and adds:

* SPARQL harvesting of REAL entity candidate pools from query.wikidata.org,
  paged with ``LIMIT``/``OFFSET`` and explicit leaf-occupation ``VALUES`` lists
  (transitive ``wdt:P279*`` closures over big human classes time the endpoint
  out, so we enumerate leaf occupations instead — see docs/dataset-v2.md).
* per-entity signal fetch: 12-month pl.wiki pageviews, wikidata sitelink count,
  plwiki article length, disambiguation flag, resolved title + QID.

Resolution is QID-based and disambiguation-safe: SPARQL already binds the QID
and the plwiki sitelink title, so we never resolve by raw title. Entities whose
plwiki article is a disambiguation page / list / redirect-to-other-type are
flagged (and can be excluded by the caller).

CPU/web only. No GPU. All network calls are checkpointed to resumable JSONL.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Iterable, Iterator

import httpx

UA = {
    "User-Agent": "bielik-hallu-research/0.3 "
    "(greg@prosit.no; paper-2 dataset-v2 build)"
}
SLEEP = 0.1
SPARQL_URL = "https://query.wikidata.org/sparql"
WIKI_API = "https://pl.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
PV_REST = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

# Last 12 full months (aligned with the P0.3 pilot window).
PV_START, PV_END = "2025070100", "2026063000"

# Wikidata QIDs.
Q_HUMAN = "Q5"
Q_POLAND = "Q36"

# Explicit leaf occupation QIDs per people domain (avoids P279* closure).
# Chosen to span each domain's sub-fields; the pool need not be exhaustive,
# only broad enough to sample 10 popularity deciles.
ATHLETE_OCCUPATIONS = [
    "Q937857",    # association football player
    "Q13381863",  # ski jumper
    "Q10833314",  # tennis player
    "Q11513337",  # athletics competitor
    "Q11338576",  # boxer
    "Q19204627",  # American football / general sportsperson variant
    "Q13382519",  # volleyball player
    "Q10843402",  # swimmer
    "Q3665646",   # basketball player
    "Q11774891",  # ice hockey / handball style
    "Q2309784",   # cyclist (sport)
    "Q13141064",  # speed skater
    "Q4009406",   # sport cyclist
    "Q12299841",  # cricketer-style catch-all (rare in PL, harmless)
]
WRITER_OCCUPATIONS = [
    "Q36180",   # writer
    "Q49757",   # poet
    "Q6625963",  # novelist
    "Q214917",  # playwright
    "Q4853732",  # children's writer
    "Q11774202",  # essayist
]
MUSICIAN_OCCUPATIONS = [
    "Q639669",  # musician
    "Q177220",  # singer
    "Q36834",   # composer
    "Q855091",  # guitarist
    "Q158852",  # conductor
    "Q486748",  # pianist
    "Q753110",  # songwriter
]
# Locality classes: cities/towns (small, fast) + village-in-Poland (the low tail).
CITY_CLASSES = ["Q515", "Q3957"]          # city, town
VILLAGE_CLASS = "Q3558970"                # village in Poland (wieś w Polsce)


# --------------------------------------------------------------------------
# Low-level HTTP (mirrors the pilot's _get).
# --------------------------------------------------------------------------
def _get(url: str, params: dict | None = None, tries: int = 4,
         timeout: float = 60.0) -> dict | None:
    for attempt in range(tries):
        try:
            r = httpx.get(url, params=params, headers=UA, timeout=timeout,
                          follow_redirects=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * 2 ** attempt)
    return None


def sparql(query: str, tries: int = 4, timeout: float = 90.0) -> list[dict]:
    """Run a SPARQL query, return the bindings list."""
    for attempt in range(tries):
        try:
            r = httpx.get(SPARQL_URL, params={"query": query, "format": "json"},
                          headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3.0 * 2 ** attempt)
    return []


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _title_from_article(url: str) -> str:
    """pl.wikipedia article URL -> human-readable page title."""
    slug = url.split("/wiki/", 1)[-1]
    return urllib.parse.unquote(slug).replace("_", " ")


# --------------------------------------------------------------------------
# SPARQL pool harvest.
# --------------------------------------------------------------------------
def _people_page_query(occupations: list[str], limit: int, offset: int) -> str:
    # No label SERVICE: it dominates query time (~30s/page) and the display
    # title comes from the plwiki article URL anyway.
    values = " ".join(f"wd:{q}" for q in occupations)
    return f"""
SELECT ?item ?article ?sitelinks WHERE {{
  VALUES ?occ {{ {values} }}
  ?item wdt:P106 ?occ ; wdt:P27 wd:{Q_POLAND} ; wdt:P31 wd:{Q_HUMAN} ;
        wikibase:sitelinks ?sitelinks .
  ?article schema:about ?item ; schema:isPartOf <https://pl.wikipedia.org/> .
}}
ORDER BY ?item LIMIT {limit} OFFSET {offset}"""


def _city_page_query(limit: int, offset: int) -> str:
    values = " ".join(f"wd:{q}" for q in CITY_CLASSES)
    return f"""
SELECT ?item ?article ?sitelinks WHERE {{
  VALUES ?cls {{ {values} }}
  ?item wdt:P31 ?cls ; wdt:P17 wd:{Q_POLAND} ; wikibase:sitelinks ?sitelinks .
  ?article schema:about ?item ; schema:isPartOf <https://pl.wikipedia.org/> .
}}
ORDER BY ?item LIMIT {limit} OFFSET {offset}"""


def _village_page_query(limit: int, offset: int) -> str:
    # Q3558970 is already "village in Poland" so P17 is redundant (and slower).
    return f"""
SELECT ?item ?article ?sitelinks WHERE {{
  ?item wdt:P31 wd:{VILLAGE_CLASS} ; wikibase:sitelinks ?sitelinks .
  ?article schema:about ?item ; schema:isPartOf <https://pl.wikipedia.org/> .
}}
ORDER BY ?item LIMIT {limit} OFFSET {offset}"""


# Query specs: domain -> list of (label, query_builder, target_rows).
# Each builder takes (limit, offset). target_rows caps paging per source.
def _domain_sources(domain: str) -> list[tuple[str, callable, int]]:
    if domain == "athletes":
        return [("athletes", lambda l, o: _people_page_query(
            ATHLETE_OCCUPATIONS, l, o), 6000)]
    if domain == "writers":
        return [("writers", lambda l, o: _people_page_query(
            WRITER_OCCUPATIONS, l, o), 6000)]
    if domain == "musicians":
        return [("musicians", lambda l, o: _people_page_query(
            MUSICIAN_OCCUPATIONS, l, o), 6000)]
    if domain == "cities":
        # cities/towns give the popular head; villages give the long low tail.
        return [
            ("cities_towns", _city_page_query, 2000),
            ("villages", _village_page_query, 6000),
        ]
    raise KeyError(domain)


def harvest_pool(domain: str, page_size: int = 500,
                 log=print) -> list[dict]:
    """Harvest the raw candidate pool for a domain from SPARQL.

    Returns dicts with: qid, title (plwiki display title), sitelinks (int),
    label (wikidata pl label if present else title), source (query label).
    De-duplicated by QID. Pageviews / article length are fetched later per
    entity (this stage is cheap SPARQL only).
    """
    seen: dict[str, dict] = {}
    for label, builder, target in _domain_sources(domain):
        offset = 0
        while offset < target:
            q = builder(page_size, offset)
            rows = sparql(q)
            if not rows:
                break
            for r in rows:
                qid = _qid(r["item"]["value"])
                if qid in seen:
                    continue
                title = _title_from_article(r["article"]["value"])
                seen[qid] = {
                    "qid": qid,
                    "title": title,
                    "label": r.get("itemLabel", {}).get("value", title),
                    "sitelinks": int(r["sitelinks"]["value"]),
                    "source": label,
                }
            log(f"  {domain}/{label}: offset={offset} +{len(rows)} "
                f"pool={len(seen)}")
            offset += page_size
            time.sleep(SLEEP)
            if len(rows) < page_size:
                break
    return list(seen.values())


# --------------------------------------------------------------------------
# Per-entity signal fetch (pageviews, article length, disambiguation).
# QID-based; title comes from SPARQL so no title-collision resolution needed.
# --------------------------------------------------------------------------
def page_info(title: str) -> dict | None:
    """Resolve a plwiki page: canonical title, length, wikibase item,
    disambiguation flag, redirect target. None if the page is missing."""
    j = _get(WIKI_API, {
        "action": "query", "format": "json", "redirects": 1,
        "titles": title, "prop": "info|pageprops",
        "ppprop": "wikibase_item|disambiguation",
    })
    pages = (j or {}).get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if int(pid) < 0 or "missing" in page:
            return None
        props = page.get("pageprops", {})
        return {
            "title": page.get("title", title),
            "length": int(page.get("length", 0)),
            "wikibase_item": props.get("wikibase_item"),
            "disambiguation": "disambiguation" in props,
        }
    return None


def pageviews(canonical_title: str) -> int:
    quoted = urllib.parse.quote(canonical_title.replace(" ", "_"), safe="")
    j = _get(f"{PV_REST}/pl.wikipedia/all-access/user/{quoted}/monthly/"
             f"{PV_START}/{PV_END}")
    if not j:
        return 0
    return int(sum(item.get("views", 0) for item in j.get("items", [])))


def sitelinks_for(qid: str | None) -> int:
    if not qid:
        return 0
    j = _get(WIKIDATA_API, {
        "action": "wbgetentities", "format": "json", "ids": qid,
        "props": "sitelinks",
    })
    ent = (j or {}).get("entities", {}).get(qid, {})
    return len(ent.get("sitelinks", {}) or {})


def fetch_signals(cand: dict) -> dict:
    """Fetch pageviews + article length + disambiguation for one candidate.

    ``cand`` must have keys: qid, title, sitelinks (from SPARQL). We trust the
    SPARQL sitelink count but refresh length/pageviews/disambiguation from the
    live page. On a missing page (rare for SPARQL-sourced titles), record zeros
    with a flag.
    """
    info = page_info(cand["title"])
    time.sleep(SLEEP)
    if info is None:
        return {
            **cand, "resolved_title": None, "article_len": 0,
            "pageviews_12m": 0, "disambiguation": False, "missing": True,
        }
    pv = pageviews(info["title"])
    time.sleep(SLEEP)
    return {
        **cand,
        "resolved_title": info["title"],
        "article_len": info["length"],
        "pageviews_12m": pv,
        "disambiguation": bool(info["disambiguation"]),
        "missing": False,
    }


# --------------------------------------------------------------------------
# Checkpointed batch signal fetch (resumable JSONL).
# --------------------------------------------------------------------------
def fetch_signals_checkpointed(
    candidates: Iterable[dict], checkpoint: Path, log=print,
) -> list[dict]:
    """Fetch signals for many candidates, resuming from a JSONL checkpoint.

    Keyed by QID. Safe to interrupt and rerun.
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["qid"]] = r
    cands = list(candidates)
    todo = [c for c in cands if c["qid"] not in done]
    log(f"signals: {len(done)} cached, {len(todo)} to fetch")
    with checkpoint.open("a") as fh:
        for i, cand in enumerate(todo):
            try:
                rec = fetch_signals(cand)
            except Exception as exc:  # log-and-continue; rerun resumes
                log(f"  ERROR {cand['qid']} {cand.get('title')}: {exc}")
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[cand["qid"]] = rec
            if (i + 1) % 50 == 0:
                log(f"  {i + 1}/{len(todo)}")
    return [done[c["qid"]] for c in cands if c["qid"] in done]


# --------------------------------------------------------------------------
# Non-existence check for fabricated names (exact + fulltext search).
# --------------------------------------------------------------------------
def plwiki_exists(name: str) -> bool:
    """True if ``name`` resolves to an existing plwiki page (redirect-aware)."""
    return page_info(name) is not None


def plwiki_fulltext_hits(name: str, limit: int = 3) -> list[str]:
    """Return plwiki fulltext-search title hits for ``name`` (quoted phrase)."""
    j = _get(WIKI_API, {
        "action": "query", "format": "json", "list": "search",
        "srsearch": f'"{name}"', "srlimit": limit, "srprop": "",
    })
    hits = (j or {}).get("query", {}).get("search", [])
    return [h["title"] for h in hits]
