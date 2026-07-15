"""Second-family judge over the v2 correctness labels (agreement and reliability).

The paper's v2 correctness labels (RQ2 behavioral mirror, RQ4 contrast (b),
risk-coverage targets) all come from a single judge (claude-opus-4-8). P0.4
only de-circularized the PREDECESSOR's v1 athletes-KNOWN verdicts (n=54, one
domain, one condition). This script closes that gap for the labels THIS paper
depends on.

Design (mirrors scripts/p0_second_judge.py exactly where it matters):
  * Re-judge a stratified ~20% sample of the 4,800 stored v2 REAL-entity answer
    verdicts (results/paper2_final/gen/<slug>/<domain>/answers.jsonl, with the
    stored Opus verdict in judge_checkpoint.jsonl).
  * All 4 final-phase models; REAL entities only (fabricated are not judged in
    the pipeline); stratified by (domain, decile): sample ceil(20%) of the 6
    entities in each (domain x decile) cell, re-judging ALL 5 answers -> per
    answer-level verdicts.
  * SAME second judge as P0.4: OpenAI gpt-5.2, EXACT build.py rubric text,
    JSON verdict, identical robust parse (unparseable -> False), checkpointed
    and resumable.

Agreement is reported at BOTH levels the paper uses:
  * answer level  (per-answer second vs Opus `correct`) — this is what the RQ4
    entity_incorrect target and contrast (b) ultimately aggregate.
  * entity level  (majority-correct >=3/5 second vs Opus) — the calibration /
    decile-correctness axis.
Cohen's kappa (same estimator as P0.4) per model + pooled, at both levels.
ALSO recomputes the Bielik-11B decile->correctness cells under the second
judge's labels on the sampled subset (does the monotone 0.06->0.68 survive?).

Outputs:
  results/paper2_final/second_judge_v2_checkpoint.jsonl   (per-answer, resumable)
  results/paper2_final/second_judge_v2.json               (agreement + kappa + decile)

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/p2_second_judge.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "paper2_final"
GEN = FINAL / "gen"
CHECKPOINT = FINAL / "second_judge_v2_checkpoint.jsonl"
OUT = FINAL / "second_judge_v2.json"

SEED = 0
SAMPLE_FRAC = 0.20
SLUGS = ("Bielik-11B-v3.0-Instruct", "gemma-4-12b-it",
         "Qwen3-14B", "PLLuM-12B-instruct-2512")
DOMAINS = ("athletes", "cities", "writers", "musicians")

PREFERENCE = ["gpt-5.2", "gpt-5.1", "gpt-5", "o4", "o3", "gpt-4.1", "gpt-4o"]
EXCLUDE_SUBSTR = ("mini", "nano", "audio", "realtime", "search", "transcribe",
                  "tts", "image", "embed", "moderation", "codex", "chat-latest",
                  "deep-research", "32k", "16k", "instruct", "preview", "pro")


def rubric(entity: str, answer: str) -> str:
    return (f"Entity: {entity}\nModel answer: {answer}\n\n"
            "Is the answer factually correct about the real world? "
            'Reply with JSON only: {"correct": true} or {"correct": false}.')


def parse_correct_verdict(text: str) -> bool:
    """Identical to build.py / p0_second_judge.py parser (unparseable -> False)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(0))
        return bool(obj["correct"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def pick_openai_model(key: str) -> str:
    r = httpx.get("https://api.openai.com/v1/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    candidates = [i for i in ids if not any(s in i for s in EXCLUDE_SUBSTR)]
    for pref in PREFERENCE:
        matches = sorted(i for i in candidates if i == pref or i.startswith(pref + "-"))
        if matches:
            return pref if pref in matches else matches[-1]
    raise RuntimeError(f"no known strong model among {sorted(candidates)[:40]}")


def openai_judge(key: str, model: str, entity: str, answer: str, tries: int = 6):
    body = {"model": model,
            "messages": [{"role": "user", "content": rubric(entity, answer)}],
            "max_completion_tokens": 2000}
    last = None
    for attempt in range(tries):
        try:
            r = httpx.post("https://api.openai.com/v1/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json=body, timeout=120)
            if r.status_code == 400 and "max_completion_tokens" in r.text:
                body.pop("max_completion_tokens", None)
                body["max_tokens"] = 200
                continue
            if r.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            return parse_correct_verdict(text), text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * 2 ** attempt)
    raise last


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Same estimator as p0_second_judge.py."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    po = float((a == b).mean())
    pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
    return (po - pe) / (1 - pe) if pe < 1.0 else float("nan")


def load_opus_verdicts():
    """(slug, uid, sample_idx) -> bool correct (Opus, from judge_checkpoint)."""
    m = {}
    for line in (FINAL / "judge_checkpoint.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m[(r["slug"], r["uid"], r["sample_idx"])] = bool(r["correct"])
    return m


def load_real_answers(slug: str):
    """list of dicts with uid, entity, domain, decile, answers[5] for REAL entities."""
    rows = []
    for dom in DOMAINS:
        p = GEN / slug / dom / "answers.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "real":
                rows.append(r)
    return rows


def stratified_sample(rows, rng):
    """Stratified ~20% entity sample.

    240 real entities/model over 40 (domain x decile) cells of 6. Take 1 entity
    per cell (guarantees full decile x domain coverage), then top up to reach
    round(20% * 240)=48 entities/model by drawing additional distinct entities
    from randomly chosen cells. -> 48 entities x 5 answers = 240 verdicts/model,
    960 total = 20.0% of 4,800.
    """
    df = pd.DataFrame(rows)
    target = int(round(SAMPLE_FRAC * len(df)))  # 48
    groups = {key: g.index.to_numpy() for key, g in df.groupby(["domain", "decile"])}
    picked = set()
    # 1 per cell
    for key, idxs in groups.items():
        picked.add(int(rng.choice(idxs)))
    # top up
    remaining = [i for g in groups.values() for i in g if int(i) not in picked]
    rng.shuffle(remaining)
    for i in remaining:
        if len(picked) >= target:
            break
        picked.add(int(i))
    return df.loc[sorted(picked)].to_dict("records")


def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("BLOCKED: no OPENAI_API_KEY. set -a; source .env; set +a", flush=True)
        sys.exit(2)
    judge_model = pick_openai_model(key)
    print(f"second judge model: {judge_model}", flush=True)

    opus = load_opus_verdicts()
    rng = np.random.default_rng(SEED)

    done = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["slug"], r["uid"], r["sample_idx"])] = r

    t0 = time.time()
    per_model = {}
    pooled_ans_second, pooled_ans_opus = [], []
    pooled_ent_second, pooled_ent_opus = [], []
    n_calls = 0
    bielik_decile_rows = []  # for the Bielik-11B monotone recheck

    with CHECKPOINT.open("a", encoding="utf-8") as fh:
        for slug in SLUGS:
            rows = load_real_answers(slug)
            sample = stratified_sample(rows, np.random.default_rng(SEED))
            ans_second, ans_opus = [], []
            ent_second, ent_opus = [], []
            for rec in sample:
                uid, entity, dec, dom = rec["uid"], rec["entity"], rec["decile"], rec["domain"]
                v_second, v_opus = [], []
                for i, a in enumerate(rec["answers"]):
                    k = (slug, uid, i)
                    if k not in done:
                        ok, raw = openai_judge(key, judge_model, entity, a)
                        d = {"slug": slug, "uid": uid, "entity": entity, "domain": dom,
                             "decile": int(dec), "sample_idx": i, "correct": bool(ok),
                             "raw": raw[:400], "judge_model": judge_model}
                        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                        fh.flush()
                        done[k] = d
                        n_calls += 1
                        time.sleep(0.2)
                    v_second.append(bool(done[k]["correct"]))
                    op = opus.get(k)
                    v_opus.append(bool(op) if op is not None else False)
                    if op is not None:
                        ans_second.append(int(bool(done[k]["correct"])))
                        ans_opus.append(int(bool(op)))
                # entity level: majority (>=3/5) correct
                e_s = int(sum(v_second) >= 3)
                e_o = int(sum(v_opus) >= 3)
                ent_second.append(e_s)
                ent_opus.append(e_o)
                if slug == "Bielik-11B-v3.0-Instruct":
                    bielik_decile_rows.append({"decile": int(dec),
                                               "second_frac": sum(v_second) / 5.0,
                                               "opus_frac": sum(v_opus) / 5.0})
            a_s = np.array(ans_second)
            a_o = np.array(ans_opus)
            e_s = np.array(ent_second)
            e_o = np.array(ent_opus)
            per_model[slug] = {
                "n_entities": int(len(e_s)), "n_answers": int(len(a_s)),
                "answer_agreement": round(float((a_s == a_o).mean()), 4),
                "answer_kappa": round(kappa(a_s, a_o), 4),
                "entity_agreement": round(float((e_s == e_o).mean()), 4),
                "entity_kappa": round(kappa(e_s, e_o), 4),
                "second_answer_positive_rate": round(float(a_s.mean()), 4),
                "opus_answer_positive_rate": round(float(a_o.mean()), 4),
            }
            pooled_ans_second.extend(ans_second)
            pooled_ans_opus.extend(ans_opus)
            pooled_ent_second.extend(ent_second)
            pooled_ent_opus.extend(ent_opus)
            print(f"[{slug}] ans_agree={per_model[slug]['answer_agreement']} "
                  f"ans_kappa={per_model[slug]['answer_kappa']} "
                  f"ent_agree={per_model[slug]['entity_agreement']} "
                  f"ent_kappa={per_model[slug]['entity_kappa']}", flush=True)

    a_s = np.array(pooled_ans_second)
    a_o = np.array(pooled_ans_opus)
    e_s = np.array(pooled_ent_second)
    e_o = np.array(pooled_ent_opus)

    # Bielik-11B decile monotone recheck (sampled subset)
    bdf = pd.DataFrame(bielik_decile_rows)
    bielik_decile = {}
    if len(bdf):
        for dec, g in bdf.groupby("decile"):
            bielik_decile[str(int(dec))] = {
                "n": int(len(g)),
                "second_correctness": round(float(g["second_frac"].mean()), 4),
                "opus_correctness": round(float(g["opus_frac"].mean()), 4),
            }

    out = {
        "meta": {
            "note": "second-family judge agreement",
            "judge_model": judge_model,
            "seed": SEED,
            "rubric": "identical text to build.py claude_judge; JSON-only verdict; "
                      "identical parse_correct_verdict fallback (unparseable -> False)",
            "sampling": f"stratified ~{int(SAMPLE_FRAC*100)}% of REAL-entity answer "
                        f"verdicts per (domain x decile) cell, all 5 answers, 4 models; "
                        f"compared against stored claude-opus-4-8 verdicts "
                        f"(judge_checkpoint.jsonl)",
            "wall_clock_s": round(time.time() - t0, 1),
            "n_api_calls_this_run": n_calls,
            "p04_v1_reference": {"agreement": 0.926, "kappa": 0.67,
                                 "note": "v1 athletes-KNOWN entity-level, n=54"},
        },
        "per_model": per_model,
        "pooled": {
            "n_answers": int(len(a_s)), "n_entities": int(len(e_s)),
            "answer_agreement": round(float((a_s == a_o).mean()), 4),
            "answer_kappa": round(kappa(a_s, a_o), 4),
            "entity_agreement": round(float((e_s == e_o).mean()), 4),
            "entity_kappa": round(kappa(e_s, e_o), 4),
            "second_answer_positive_rate": round(float(a_s.mean()), 4),
            "opus_answer_positive_rate": round(float(a_o.mean()), 4),
        },
        "bielik11b_decile_recheck": bielik_decile,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    print(f"POOLED answer agreement={out['pooled']['answer_agreement']} "
          f"kappa={out['pooled']['answer_kappa']}; "
          f"entity agreement={out['pooled']['entity_agreement']} "
          f"kappa={out['pooled']['entity_kappa']}; n_api_calls={n_calls}")


if __name__ == "__main__":
    main()
