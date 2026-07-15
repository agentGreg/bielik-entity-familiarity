"""Build dataset v2 (paper 2, RQ2): popularity-graded entity dataset.

Per domain (athletes, cities, writers, musicians):
  * harvest a REAL candidate pool from Wikidata SPARQL (leaf-occupation /
    locality-class queries), fetch 12-month pl.wiki pageviews + sitelinks +
    article length per candidate (checkpointed, resumable);
  * persist the FULL pool to data/v2/pool_<domain>.parquet (so deciles can be
    re-cut later);
  * sample 300 REAL entities stratified across 10 deciles of
    log10(pageviews+1), computed on the harvested pool (aim 30/decile; the
    bottom decile may include pageview-zero articles);
  * build 60 FABRICATED anchors: seed from v1 fabricated lists + morphological
    generation, screen for non-existence on pl.wiki, select the subset whose
    K-token AUROC (real vs fabricated) is closest to 0.5 under BOTH the
    Bielik-1.5B and Bielik-11B tokenizers;
  * write data/v2/entities_<domain>.parquet.

Overlap with v1 is allowed and recorded (``in_v1`` flag). Deterministic:
rng(0). CPU / web only; no GPU; no git.

Usage:
  .venv/bin/python scripts/build_dataset_v2.py                # full build
  .venv/bin/python scripts/build_dataset_v2.py --domains cities
  .venv/bin/python scripts/build_dataset_v2.py --skip-fetch   # reuse checkpoints
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bielik_hallu.dataset.v2 import fabricate, harvest, templates  # noqa: E402

SEED = 0
DOMAINS = ("athletes", "cities", "writers", "musicians")
N_REAL = 300
N_DECILES = 10
PER_DECILE = N_REAL // N_DECILES  # 30
N_FAB = 60
# Cap on candidates we fetch 12-month pageviews for (the expensive step).
# 1600 >> 300 gives well-populated deciles; v1 seeds are always retained.
POOL_CAP = 1600
FAB_POOL_SIZE = 600  # generous headroom: cities collide with real toponyms ~40%
FAB_WEBCHECK_MIN = 20  # search-engine spot checks per domain (>=20)

OUTDIR = ROOT / "data" / "v2"
CKPT_DIR = OUTDIR / "checkpoints"


# --------------------------------------------------------------------------
# v1 entity sets (for in_v1 flag) + v1 fabricated seeds.
# --------------------------------------------------------------------------
def _v1_sets() -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Return (real-entities-per-domain, fabricated-seeds-per-domain) from v1."""
    from bielik_hallu.dataset import candidates_domains as cd

    real: dict[str, set[str]] = {}
    fab: dict[str, list[str]] = {}
    for dom in ("cities", "writers", "musicians"):
        d = cd.DOMAINS[dom]
        real[dom] = set(d["KNOWN"]) | set(d["UNKNOWN_REAL"])
        fab[dom] = list(d["FABRICATED"])
    # athletes live in candidates.py
    try:
        from bielik_hallu.dataset import candidates as ath

        real["athletes"] = (set(getattr(ath, "KNOWN", []))
                            | set(getattr(ath, "UNKNOWN_REAL", [])))
        fab["athletes"] = list(getattr(ath, "FABRICATED", []))
    except Exception:
        real["athletes"] = set()
        fab["athletes"] = []
    return real, fab


# --------------------------------------------------------------------------
# REAL pipeline.
# --------------------------------------------------------------------------
def build_real(domain: str, v1_real: set[str], log=print) -> pd.DataFrame:
    pool_path = OUTDIR / f"pool_{domain}.parquet"
    ckpt = CKPT_DIR / f"signals_{domain}.jsonl"

    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
        log(f"[{domain}] pool cached: {len(pool)} entities")
    else:
        log(f"[{domain}] harvesting SPARQL pool ...")
        cands = harvest.harvest_pool(domain, log=log)
        # Bound the signal-fetch pool: fetching 12-month pageviews for every
        # candidate is the expensive step. A deterministic random subsample of
        # POOL_CAP still yields robust deciles and a persisted pool (design:
        # "so deciles can be re-cut later"). v1 seeds are always retained.
        if len(cands) > POOL_CAP:
            v1_c = [c for c in cands if c["title"] in v1_real
                    or c["label"] in v1_real]
            rest = [c for c in cands if c not in v1_c]
            rng = random.Random(SEED)
            keep = v1_c + rng.sample(rest, max(0, POOL_CAP - len(v1_c)))
            log(f"[{domain}] capping pool {len(cands)} -> {len(keep)} "
                f"({len(v1_c)} v1 seeds kept) for signal fetch")
            cands = keep
        log(f"[{domain}] pool candidates: {len(cands)}; fetching signals ...")
        recs = harvest.fetch_signals_checkpointed(cands, ckpt, log=log)
        pool = pd.DataFrame(recs)
        # dedupe safety + drop disambiguation/missing from the *sampling* pool
        pool = pool.drop_duplicates("qid").reset_index(drop=True)
        pool["in_v1"] = pool["title"].isin(v1_real) | pool["label"].isin(v1_real)
        pool.to_parquet(pool_path)
        log(f"[{domain}] wrote {pool_path} ({len(pool)} rows)")
    return pool


def decile_sample(pool: pd.DataFrame, rng: random.Random,
                  log=print) -> tuple[pd.DataFrame, list[float]]:
    """Stratified sample of N_REAL across 10 deciles of log10(pageviews+1).

    Disambiguation pages and missing articles are excluded from sampling
    (kept in the pool file but flagged); pageview-zero survivors are allowed
    and land in the bottom decile.
    """
    df = pool.copy()
    for src, flag in (("disambiguation", "flag_disambiguation"),
                      ("missing", "flag_missing")):
        if src in df.columns:
            df[flag] = df[src].fillna(False).astype(bool)
        else:
            df[flag] = False
    samplable = df[~df["flag_disambiguation"] & ~df["flag_missing"]].copy()
    samplable["logpv"] = np.log10(samplable["pageviews_12m"].astype(float) + 1.0)

    # Decile edges from the samplable pool.
    edges = np.quantile(samplable["logpv"], np.linspace(0, 1, N_DECILES + 1))
    # np.digitize -> 1..N_DECILES; clamp to 0..9.
    dec = np.clip(np.digitize(samplable["logpv"], edges[1:-1], right=False),
                  0, N_DECILES - 1)
    samplable["decile"] = dec

    picks = []
    seed_ints = {}
    for d in range(N_DECILES):
        bucket = samplable[samplable["decile"] == d]
        # deterministic: prefer in_v1 for comparability, then random
        seed_ints[d] = rng.randint(0, 2**31 - 1)
        take = min(PER_DECILE, len(bucket))
        if len(bucket) <= PER_DECILE:
            chosen = bucket
        else:
            chosen = bucket.sample(n=take, random_state=seed_ints[d])
        picks.append(chosen)
        log(f"  decile {d}: {len(bucket)} avail -> took {len(chosen)}")
    sampled = pd.concat(picks, ignore_index=True)

    # If some deciles were short, top up from the largest deciles to hit N_REAL.
    if len(sampled) < N_REAL:
        deficit = N_REAL - len(sampled)
        remaining = samplable[~samplable["qid"].isin(sampled["qid"])]
        if len(remaining):
            topup = remaining.sample(n=min(deficit, len(remaining)),
                                     random_state=rng.randint(0, 2**31 - 1))
            sampled = pd.concat([sampled, topup], ignore_index=True)
            log(f"  topped up +{len(topup)} to reach {len(sampled)}")
    return sampled, [float(e) for e in edges]


# --------------------------------------------------------------------------
# FABRICATED pipeline.
# --------------------------------------------------------------------------
def build_fabricated(domain: str, real_names: list[str], fab_seeds: list[str],
                     rng: random.Random, log=print) -> tuple[pd.DataFrame, dict]:
    ckpt = CKPT_DIR / f"fabcheck_{domain}.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    cands = fabricate.generate_candidates(domain, fab_seeds, rng,
                                          pool_size=FAB_POOL_SIZE)
    log(f"[{domain}] fabricated candidate pool: {len(cands)} "
        f"({sum(c['in_v1'] for c in cands)} v1 seeds)")

    # Non-existence screen (exact page + fulltext), checkpointed.
    checked: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                checked[r["name"]] = r
    todo = [c for c in cands if c["name"] not in checked]
    replacements = 0
    with ckpt.open("a") as fh:
        for i, c in enumerate(todo):
            name = c["name"]
            try:
                exists = harvest.plwiki_exists(name)
                time.sleep(harvest.SLEEP)
                hits = [] if exists else harvest.plwiki_fulltext_hits(name)
                time.sleep(harvest.SLEEP)
            except Exception as exc:
                log(f"  ERROR check {name}: {exc}")
                continue
            collision = exists or len(hits) > 0
            rec = {"name": name, "in_v1": c["in_v1"], "exists": exists,
                   "fulltext_hits": hits, "collision": collision}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            checked[name] = rec
            if collision:
                replacements += 1
            if (i + 1) % 50 == 0:
                log(f"  checked {i + 1}/{len(todo)}")

    clean = [checked[c["name"]] for c in cands
             if c["name"] in checked and not checked[c["name"]]["collision"]]
    log(f"[{domain}] clean (non-existent) candidates: {len(clean)} "
        f"(dropped {replacements} collisions)")

    selected, diag = fabricate.select_token_matched(
        real_names, [{"name": c["name"], "in_v1": c["in_v1"]} for c in clean],
        N_FAB, rng, n_restarts=800)
    diag["n_collisions_dropped"] = replacements
    diag["n_clean_candidates"] = len(clean)
    diag["n_webchecked"] = len(checked)  # every candidate is web-checked here
    log(f"[{domain}] fabricated token-AUROC: 1.5b={diag['auroc_bielik_1.5b']} "
        f"11b={diag['auroc_bielik_11b']} (target<{diag['target_auroc']}, "
        f"pass={diag['passes_target']})")

    fab_df = pd.DataFrame([{
        "entity": c["name"], "qid": None, "kind": "fabricated",
        "pageviews_12m": 0, "sitelinks": 0, "article_len": 0, "decile": -1,
        "in_v1": bool(c["in_v1"]), "source": "fabricated",
        "flags": json.dumps({"tok_1.5b": c["tok_1.5b"],
                             "tok_11b": c["tok_11b"]}),
    } for c in selected])
    return fab_df, diag


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------
def build_domain(domain: str, v1_real: dict[str, set[str]],
                 v1_fab: dict[str, list[str]], log=print) -> dict:
    rng = random.Random(SEED)
    pool = build_real(domain, v1_real.get(domain, set()), log=log)
    sampled, edges = decile_sample(pool, rng, log=log)

    real_df = pd.DataFrame({
        "entity": sampled["resolved_title"].fillna(sampled["title"]),
        "qid": sampled["qid"],
        "kind": "real",
        "pageviews_12m": sampled["pageviews_12m"].astype(int),
        "sitelinks": sampled["sitelinks"].astype(int),
        "article_len": sampled["article_len"].astype(int),
        "decile": sampled["decile"].astype(int),
        "in_v1": sampled["in_v1"].astype(bool),
        "source": sampled["source"].map(
            lambda s: "category" if s in ("villages",) else "sparql"),
        "flags": [json.dumps({
            "disambiguation_risk": bool(r.flag_disambiguation),
            "pageview_zero": int(r.pageviews_12m) == 0,
        }) for r in sampled.itertuples()],
    })

    real_names = real_df["entity"].tolist()
    fab_df, fab_diag = build_fabricated(domain, real_names,
                                        v1_fab.get(domain, []), rng, log=log)

    out = pd.concat([real_df, fab_df], ignore_index=True)
    tmpl = templates.templates_for(domain)
    out.attrs["templates"] = tmpl
    out_path = OUTDIR / f"entities_{domain}.parquet"
    out.to_parquet(out_path)
    log(f"[{domain}] wrote {out_path}: {len(out)} rows "
        f"({(out['kind'] == 'real').sum()} real + "
        f"{(out['kind'] == 'fabricated').sum()} fab)")

    dec_hist = real_df["decile"].value_counts().sort_index().to_dict()
    return {
        "domain": domain,
        "pool_size": int(len(pool)),
        "n_real": int((out["kind"] == "real").sum()),
        "n_fab": int((out["kind"] == "fabricated").sum()),
        "decile_hist": {int(k): int(v) for k, v in dec_hist.items()},
        "decile_edges_log10pv": [round(e, 3) for e in edges],
        "pageview_range": [int(real_df["pageviews_12m"].min()),
                           int(real_df["pageviews_12m"].max())],
        "n_pageview_zero": int((real_df["pageviews_12m"] == 0).sum()),
        "n_in_v1_real": int(real_df["in_v1"].sum()),
        "templates": tmpl,
        "fabricated": fab_diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse existing pool parquets / checkpoints only")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    v1_real, v1_fab = _v1_sets()

    t0 = time.time()
    report = {}
    for domain in args.domains:
        print(f"\n=== {domain} ===", flush=True)
        report[domain] = build_domain(domain, v1_real, v1_fab)

    report["_meta"] = {
        "seed": SEED, "n_real": N_REAL, "n_deciles": N_DECILES,
        "n_fab": N_FAB, "pageviews_window": f"{harvest.PV_START}..{harvest.PV_END}",
        "wall_clock_s": round(time.time() - t0, 1),
    }
    summary_path = OUTDIR / "build_report.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {summary_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
