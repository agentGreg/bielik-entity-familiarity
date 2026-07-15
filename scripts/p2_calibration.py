"""Paper-2 final phase, step 2: behavioral labels (strict judge) + refusals.

For every REAL entity in the eval subset and every final-phase model, judge
all 5 sampled answers with the strict rubric from
``bielik_hallu.dataset.build.claude_judge`` (claude-opus-4-8, verbatim prompt)
and derive the per-entity correctness rate (0-5 of 5). Fabricated entities
are NOT judged: any confident (non-refusal) answer to a fabricated entity is
a confabulation by construction. Refusal markers are counted on EVERYTHING
(real + fabricated) using the paper's 6-substring Polish heuristic plus a
small documented extension set for non-Polish-tuned families (Gemma/Qwen
answer PL questions but sometimes refuse in English or with different
phrasing); base and extended counts are reported separately.

Outputs (all under results/paper2_final/):
  judge_checkpoint.jsonl     per-answer verdicts, keyed (slug, qid, sample_idx)
  behavioral_labels.parquet  per (slug, qid): n_correct, n_refusal_*,
                             majority_correct, entity_incorrect (RQ4 target)
  calibration.json           decile -> mean correctness curve per model
                             (pooled + per domain), refusal counts per
                             model x kind, judge metadata

Entity-level RQ4 target ``entity_incorrect``:
  real entity        -> 1 if fewer than 3 of 5 answers judged correct
  fabricated entity  -> 1 if fewer than 3 of 5 answers are refusals
                        (majority-confabulation; a refusal is the only
                        correct behavior for a non-existent entity)

Usage:
    set -a; source .env; set +a
    uv run python scripts/p2_calibration.py [--models slug1,slug2]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd

N_WORKERS = 8  # concurrent judge calls (checkpoint writes are lock-guarded)
_CKPT_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "paper2_final"
GEN_ROOT = OUT_ROOT / "gen"
CHECKPOINT = OUT_ROOT / "judge_checkpoint.jsonl"

JUDGE_MODEL = "claude-opus-4-8"  # same family+rubric as dataset build.py

SLUGS = (
    "Bielik-11B-v3.0-Instruct",
    "gemma-4-12b-it",
    "Qwen3-14B",
    "PLLuM-12B-instruct-2512",
)
DOMAINS = ("athletes", "cities", "writers", "musicians")

# Mirrors config.REFUSAL_MARKERS exactly (kept in sync manually; scripts stay
# decoupled from src/ mutations per ownership rules).
REFUSAL_MARKERS = (
    "nie wiem", "nie znam", "nie mam informacji", "brak informacji",
    "nie jestem w stanie", "nie posiadam",
)
# Documented extension for cross-family refusal phrasing (EN + extra PL).
REFUSAL_MARKERS_EXT = REFUSAL_MARKERS + (
    "nie mam danych", "nie mogę znaleźć", "nie mogę podać", "nie udało mi się",
    "nie ma informacji", "nie znalazłem", "nie znalazłam", "nie kojarzę",
    "i don't know", "i do not know", "i'm not aware", "i am not aware",
    "i don't have information", "i do not have information",
    "i couldn't find", "i could not find", "no information",
    "not a well-known", "unable to find", "i'm not familiar", "i am not familiar",
)


def heuristic_refusal(answer: str, markers=REFUSAL_MARKERS) -> bool:
    low = answer.lower()
    return any(m in low for m in markers)


def parse_correct_verdict(text: str) -> bool:
    """Mirrors bielik_hallu.dataset.build.parse_correct_verdict."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text!r}")
    obj = json.loads(match.group(0))
    return bool(obj["correct"])


def call_with_retry(fn, tries: int = 8, base_delay: float = 2.0, sleep=time.sleep):
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad for retry
            last_exc = exc
            if attempt < tries - 1:
                sleep(base_delay * 2**attempt)
    raise last_exc


def claude_judge(client: anthropic.Anthropic, entity: str, answer: str) -> bool:
    """Verbatim rubric from bielik_hallu.dataset.build.claude_judge."""
    def _do():
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Entity: {entity}\nModel answer: {answer}\n\n"
                    "Is the answer factually correct about the real world? "
                    'Reply with JSON only: {"correct": true} or {"correct": false}.'
                ),
            }],
        )
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise ValueError(f"no text block: {msg.content!r}")
        return parse_correct_verdict(text)
    return call_with_retry(_do)


def load_checkpoint(path: Path) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[(rec["slug"], rec["uid"], rec["sample_idx"])] = rec
    return done


def append_checkpoint(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def load_answers(slug: str) -> list[dict]:
    rows = []
    for domain in DOMAINS:
        p = GEN_ROOT / slug / domain / "answers.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"missing generation output: {p}")
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _judge_one(client: anthropic.Anthropic, slug: str, uid: str,
               entity: str, sample_idx: int, answer: str,
               done: dict[tuple, dict]) -> None:
    try:
        ok = claude_judge(client, entity, answer)
        failed = False
    except Exception as exc:  # noqa: BLE001 - exhausted retries
        print(f"  JUDGE FAIL {slug}/{entity}/{sample_idx}: {exc}", flush=True)
        ok, failed = False, True
    crec = {"slug": slug, "uid": uid, "entity": entity,
            "sample_idx": sample_idx, "correct": ok, "parse_failed": failed}
    with _CKPT_LOCK:
        append_checkpoint(CHECKPOINT, crec)
        done[(slug, uid, sample_idx)] = crec


def process_slug(client: anthropic.Anthropic, slug: str,
                 done: dict[tuple, dict]) -> tuple[list[dict], int]:
    entity_rows: list[dict] = []
    records = load_answers(slug)
    print(f"[calibration] {slug}: {len(records)} entities", flush=True)

    # Phase 1: fire all pending judge calls concurrently (checkpointed).
    pending = []
    for rec in records:
        if rec["kind"] != "real":
            continue
        for i, ans in enumerate(rec["answers"]):
            if (slug, rec["uid"], i) not in done:
                pending.append((rec["uid"], rec["entity"], i, ans))
    n_calls = len(pending)
    if pending:
        print(f"[calibration] {slug}: {n_calls} judge calls "
              f"({N_WORKERS} workers)", flush=True)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = [pool.submit(_judge_one, client, slug, uid, ent, i, ans, done)
                    for uid, ent, i, ans in pending]
            for k, f in enumerate(futs):
                f.result()
                if (k + 1) % 200 == 0:
                    print(f"  [judge] {k + 1}/{n_calls}", flush=True)

    # Phase 2: assemble entity rows from the checkpoint map.
    for rec in records:
        uid, entity, kind = rec["uid"], rec["entity"], rec["kind"]
        answers = rec["answers"]
        if kind == "real":
            verdicts = [bool(done[(slug, uid, i)]["correct"])
                        for i in range(len(answers))]
        else:
            verdicts = [None] * len(answers)

        n_ref = sum(heuristic_refusal(a) for a in answers)
        n_ref_ext = sum(heuristic_refusal(a, REFUSAL_MARKERS_EXT) for a in answers)
        n_correct = sum(bool(v) for v in verdicts) if kind == "real" else None
        if kind == "real":
            entity_incorrect = int(n_correct < 3)
            majority_correct = int(n_correct >= 3)
        else:
            entity_incorrect = int(n_ref_ext < 3)
            majority_correct = None
        entity_rows.append({
            "slug": slug, "domain": rec["domain"], "uid": uid,
            "entity": entity, "kind": kind, "decile": rec["decile"],
            "n_correct": n_correct, "n_refusal": n_ref,
            "n_refusal_ext": n_ref_ext,
            "majority_correct": majority_correct,
            "entity_incorrect": entity_incorrect,
        })
    return entity_rows, n_calls


def build_calibration_summary(df: pd.DataFrame) -> dict:
    out: dict = {"judge_model": JUDGE_MODEL, "per_model": {}}
    for slug, g in df.groupby("slug"):
        real = g[g["kind"] == "real"]
        fab = g[g["kind"] != "real"]
        decile_curve = {
            str(dec): round(float(gg["n_correct"].mean()) / 5.0, 4)
            for dec, gg in real.groupby("decile")
        }
        per_domain = {}
        for dom, gd in real.groupby("domain"):
            per_domain[dom] = {
                str(dec): round(float(gg["n_correct"].mean()) / 5.0, 4)
                for dec, gg in gd.groupby("decile")
            }
        out["per_model"][slug] = {
            "decile_correctness": decile_curve,
            "decile_correctness_per_domain": per_domain,
            "real_majority_correct_rate": round(float(real["majority_correct"].mean()), 4),
            "real_mean_correct_of5": round(float(real["n_correct"].mean()), 3),
            "refusals": {
                "real_answers_refused_base": int(real["n_refusal"].sum()),
                "real_answers_refused_ext": int(real["n_refusal_ext"].sum()),
                "fab_answers_refused_base": int(fab["n_refusal"].sum()),
                "fab_answers_refused_ext": int(fab["n_refusal_ext"].sum()),
                "n_real_answers": int(len(real)) * 5,
                "n_fab_answers": int(len(fab)) * 5,
                "fab_entities_majority_refusal": int((fab["n_refusal_ext"] >= 3).sum()),
                "n_fab_entities": int(len(fab)),
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated slugs (default: all four)")
    args = ap.parse_args()
    slugs = tuple(args.models.split(",")) if args.models else SLUGS

    client = anthropic.Anthropic()
    done = load_checkpoint(CHECKPOINT)
    all_rows: list[dict] = []
    total_calls = 0
    for slug in slugs:
        rows, n_calls = process_slug(client, slug, done)
        all_rows.extend(rows)
        total_calls += n_calls
        print(f"[calibration] {slug}: {n_calls} judge calls this run", flush=True)

    df = pd.DataFrame(all_rows)
    out_parquet = OUT_ROOT / "behavioral_labels.parquet"
    # Merge with any previously written slugs not in this run.
    if out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        prev = prev[~prev["slug"].isin(df["slug"].unique())]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out_parquet)
    print(f"[calibration] wrote {out_parquet} ({len(df)} rows)", flush=True)

    summary = build_calibration_summary(df)
    summary["n_api_calls_this_run"] = total_calls
    with (OUT_ROOT / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[calibration] wrote {OUT_ROOT / 'calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
