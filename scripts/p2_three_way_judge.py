"""LLM three-way judge (refuse/correct/incorrect).

We complement the binary judge with a three-way LLM judge (refuse / correct /
incorrect). The binary judge scores explicit refusals as ``correct`` ~88% of
the time and so conflates abstention with factual accuracy for the one
abstaining model (Gemma).

A prior pass (``p2_refusal_separation.py``) worked around this with a
DETERMINISTIC Polish marker list and produced the numbers now in the paper
(Gemma 533/1200 real refusals, answered-only base error 0.90, only 13 of 149
answered entities majority-correct). This script runs the actual three-way
LLM judge and validates those marker-based numbers judge-natively.

Judge: claude-opus-4-8 (same model/family as the binary judge), verbatim
three-way rubric below. REAL entities only (fabricated aren't factually
judgeable). 240 real entities x 5 answers x 4 models = 4,800 judgments.

Checkpointed to results/paper2_robustness/three_way_judge_checkpoint.jsonl,
keyed (slug, uid, sample_idx) so reruns skip completed calls.

Outputs:
  three_way_judge_checkpoint.jsonl   per-answer verdicts
  three_way_judge.json / .md         per-model distribution, marker-vs-LLM
                                     refusal agreement, judge-native recompute
                                     of the marker-based quantities

Usage:
    uv run python scripts/p2_three_way_judge.py [--smoke] [--models a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

# --- env: load ANTHROPIC_API_KEY from .env -------------------------------
ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - dotenv optional; fall back to manual parse
    envp = ROOT / ".env"
    if envp.exists():
        for line in envp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

N_WORKERS = 8
_CKPT_LOCK = threading.Lock()

FINAL = ROOT / "results" / "paper2_final"
GEN_ROOT = FINAL / "gen"
OUT = ROOT / "results" / "paper2_robustness"
CHECKPOINT = OUT / "three_way_judge_checkpoint.jsonl"

JUDGE_MODEL = "claude-opus-4-8"  # same family+rubric family as the binary judge

SLUGS = (
    "Bielik-11B-v3.0-Instruct",
    "gemma-4-12b-it",
    "Qwen3-14B",
    "PLLuM-12B-instruct-2512",
)
SHORT = {
    "Bielik-11B-v3.0-Instruct": "Bielik-11B",
    "gemma-4-12b-it": "Gemma-12B",
    "Qwen3-14B": "Qwen-14B",
    "PLLuM-12B-instruct-2512": "PLLuM-12B",
}
DOMAINS = ("athletes", "cities", "writers", "musicians")

# Published marker list (same 6 substrings as p2_refusal_separation.py / paper).
MARK = ["nie wiem", "nie mam informacji", "nie posiadam",
        "nie jestem w stanie", "nie znam", "brak informacji"]
VALID = {"refuse", "correct", "incorrect"}


def marker_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in MARK)


def parse_verdict(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text!r}")
    obj = json.loads(match.group(0))
    v = str(obj["verdict"]).strip().lower()
    if v not in VALID:
        raise ValueError(f"verdict not in {VALID}: {v!r}")
    return v


def call_with_retry(fn, tries: int = 8, base_delay: float = 2.0, sleep=time.sleep):
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad for retry
            last_exc = exc
            if attempt < tries - 1:
                sleep(base_delay * 2 ** attempt)
    raise last_exc


def three_way_judge(client: anthropic.Anthropic, entity: str, answer: str) -> str:
    def _do():
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Entity: {entity}\nModel answer: {answer}\n\n"
                    'Classify the answer as exactly one of: "refuse" (the model '
                    'declines/says it doesn\'t know), "correct" (a factual claim '
                    'that is correct about the real world), or "incorrect" (a '
                    "factual claim that is wrong). Reply with JSON only: "
                    '{"verdict": "refuse"|"correct"|"incorrect"}.'
                ),
            }],
        )
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise ValueError(f"no text block: {msg.content!r}")
        return parse_verdict(text)
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


def _judge_one(client, slug, uid, entity, sample_idx, answer, done) -> None:
    try:
        verdict = three_way_judge(client, entity, answer)
        failed = False
    except Exception as exc:  # noqa: BLE001 - exhausted retries
        print(f"  JUDGE FAIL {slug}/{entity}/{sample_idx}: {exc}", flush=True)
        verdict, failed = "incorrect", True
    crec = {"slug": slug, "uid": uid, "entity": entity,
            "sample_idx": sample_idx, "verdict": verdict,
            "marker_refuse": marker_refusal(answer), "parse_failed": failed}
    with _CKPT_LOCK:
        append_checkpoint(CHECKPOINT, crec)
        done[(slug, uid, sample_idx)] = crec


def run_smoke(client) -> None:
    """5-answer smoke test on one Gemma entity; print verdicts, no checkpoint."""
    recs = [r for r in load_answers("gemma-4-12b-it") if r["kind"] == "real"]
    # pick an entity that mixes refusals and content if possible
    rec = recs[0]
    for r in recs:
        vals = [marker_refusal(a) for a in r["answers"]]
        if any(vals) and not all(vals):
            rec = r
            break
    print(f"[smoke] entity={rec['entity']} uid={rec['uid']} "
          f"domain={rec['domain']} decile={rec['decile']}")
    for i, ans in enumerate(rec["answers"]):
        v = three_way_judge(client, rec["entity"], ans)
        print(f"  sample {i}: verdict={v:10s} marker_refuse={marker_refusal(ans)} "
              f"| {ans[:90]!r}")


def judge_all(client, slugs, done) -> int:
    total = 0
    for slug in slugs:
        records = [r for r in load_answers(slug) if r["kind"] == "real"]
        pending = []
        for rec in records:
            for i, ans in enumerate(rec["answers"]):
                if (slug, rec["uid"], i) not in done:
                    pending.append((rec["uid"], rec["entity"], i, ans))
        n = len(pending)
        total += n
        print(f"[judge] {slug}: {len(records)} real entities, "
              f"{n} pending calls ({N_WORKERS} workers)", flush=True)
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = [pool.submit(_judge_one, client, slug, uid, ent, i, ans, done)
                    for uid, ent, i, ans in pending]
            for k, f in enumerate(futs):
                f.result()
                if (k + 1) % 200 == 0:
                    print(f"  [judge] {slug} {k + 1}/{n}", flush=True)
    return total


# --- analysis ------------------------------------------------------------

def analyze(slugs, done) -> dict:
    summary: dict = {"judge_model": JUDGE_MODEL, "marker_list": MARK, "per_model": {}}
    for slug in slugs:
        records = [r for r in load_answers(slug) if r["kind"] == "real"]
        # per-answer table
        per_ans = []
        for rec in records:
            for i, ans in enumerate(rec["answers"]):
                key = (slug, rec["uid"], i)
                if key not in done:
                    continue
                per_ans.append({
                    "uid": rec["uid"], "domain": rec["domain"],
                    "decile": rec["decile"], "sample_idx": i,
                    "verdict": done[key]["verdict"],
                    "marker_refuse": marker_refusal(ans),
                })
        n = len(per_ans)
        n_ref = sum(a["verdict"] == "refuse" for a in per_ans)
        n_cor = sum(a["verdict"] == "correct" for a in per_ans)
        n_inc = sum(a["verdict"] == "incorrect" for a in per_ans)
        n_answered = n_cor + n_inc
        # marker vs LLM refuse agreement
        agree = sum((a["verdict"] == "refuse") == a["marker_refuse"] for a in per_ans)
        both = sum((a["verdict"] == "refuse") and a["marker_refuse"] for a in per_ans)
        llm_only = sum((a["verdict"] == "refuse") and not a["marker_refuse"]
                       for a in per_ans)
        marker_only = sum((a["verdict"] != "refuse") and a["marker_refuse"]
                          for a in per_ans)
        n_marker_ref = sum(a["marker_refuse"] for a in per_ans)

        # entity-level: refused entity if majority (>=3/5) refused; answered otherwise
        by_ent: dict[str, list] = {}
        by_ent_dec: dict[str, int] = {}
        for a in per_ans:
            by_ent.setdefault(a["uid"], []).append(a)
            by_ent_dec[a["uid"]] = a["decile"]
        n_answered_entities = 0
        n_majcorrect_answered = 0
        n_majcorrect_all = 0
        for uid, samples in by_ent.items():
            nref = sum(s["verdict"] == "refuse" for s in samples)
            ncor = sum(s["verdict"] == "correct" for s in samples)
            tot = len(samples)
            maj_correct = ncor >= (tot // 2 + 1)  # >=3 of 5
            if maj_correct:
                n_majcorrect_all += 1
            if nref < (tot // 2 + 1):  # not majority-refused -> answered entity
                n_answered_entities += 1
                if maj_correct:
                    n_majcorrect_answered += 1

        # per-decile answered-correctness (answered samples only)
        dec_ans: dict[int, list] = {}
        for a in per_ans:
            if a["verdict"] != "refuse":
                dec_ans.setdefault(a["decile"], []).append(a["verdict"] == "correct")
        per_decile = {str(d): round(sum(v) / len(v), 4)
                      for d, v in sorted(dec_ans.items()) if v}

        base_err_answered = round(1 - n_cor / n_answered, 4) if n_answered else None
        summary["per_model"][SHORT[slug]] = {
            "slug": slug,
            "n_answers": n,
            "refuse": n_ref, "correct": n_cor, "incorrect": n_inc,
            "refusal_rate_llm": round(n_ref / n, 4) if n else None,
            "marker_refusals": n_marker_ref,
            "marker_refusal_rate": round(n_marker_ref / n, 4) if n else None,
            "marker_vs_llm_agreement": round(agree / n, 4) if n else None,
            "refuse_both": both, "refuse_llm_only": llm_only,
            "refuse_marker_only": marker_only,
            "n_answered_samples": n_answered,
            "correct_given_answered": round(n_cor / n_answered, 4) if n_answered else None,
            "base_err_answered": base_err_answered,
            "base_err_all_incl_refuse_as_wrong": round(1 - n_cor / n, 4) if n else None,
            "n_entities": len(by_ent),
            "n_answered_entities": n_answered_entities,
            "n_majority_correct_answered": n_majcorrect_answered,
            "n_majority_correct_all_entities": n_majcorrect_all,
            "per_decile_answered_correct": per_decile,
        }
    return summary


def write_md(summary: dict) -> str:
    pm = summary["per_model"]
    order = ["Bielik-11B", "Gemma-12B", "Qwen-14B", "PLLuM-12B"]
    order = [m for m in order if m in pm]

    def row(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines = [
        "# LLM three-way judge (refuse / correct / incorrect)\n",
        f"Judge: `{summary['judge_model']}`, verbatim three-way rubric. REAL entities "
        "only. This is the three-way LLM judge; it "
        "validates the marker-based separation already in the paper.\n",
        "## Three-way distribution per model\n",
        row(["model", "answers", "refuse", "correct", "incorrect",
             "refusal_rate", "correct|answered", "base_err_answered"]),
        row(["---"] * 8),
    ]
    for m in order:
        d = pm[m]
        lines.append(row([m, d["n_answers"], d["refuse"], d["correct"],
                          d["incorrect"], d["refusal_rate_llm"],
                          d["correct_given_answered"], d["base_err_answered"]]))
    lines += [
        "\n## Marker-list vs. LLM-`refuse` agreement (per answer)\n",
        row(["model", "LLM refuse", "marker refuse", "agreement",
             "both", "LLM-only", "marker-only"]),
        row(["---"] * 7),
    ]
    for m in order:
        d = pm[m]
        lines.append(row([m, d["refuse"], d["marker_refusals"],
                          d["marker_vs_llm_agreement"], d["refuse_both"],
                          d["refuse_llm_only"], d["refuse_marker_only"]]))
    lines += [
        "\n## Entity-level (judge-native)\n",
        row(["model", "entities", "answered entities",
             "maj-correct | answered", "maj-correct (all)"]),
        row(["---"] * 5),
    ]
    for m in order:
        d = pm[m]
        lines.append(row([m, d["n_entities"], d["n_answered_entities"],
                          d["n_majority_correct_answered"],
                          d["n_majority_correct_all_entities"]]))
    lines.append("\n## Per-decile answered-correctness (answered samples only)\n")
    for m in order:
        lines.append(f"- **{m}**: {pm[m]['per_decile_answered_correct']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run 5-answer smoke test on one Gemma entity and exit")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated slugs (default: all four)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip judging; just (re)build json/md from checkpoint")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (checked env + .env). STOP.")
    slugs = tuple(args.models.split(",")) if args.models else SLUGS

    client = anthropic.Anthropic()
    if args.smoke:
        run_smoke(client)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(CHECKPOINT)
    if not args.analyze_only:
        n = judge_all(client, slugs, done)
        print(f"[judge] {n} new judge calls this run", flush=True)
        done = load_checkpoint(CHECKPOINT)

    summary = analyze(slugs, done)
    (OUT / "three_way_judge.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "three_way_judge.md").write_text(write_md(summary), encoding="utf-8")
    print(f"[out] wrote {OUT / 'three_way_judge.json'} and .md", flush=True)
    # console summary
    for m, d in summary["per_model"].items():
        print(f"  {m}: refuse={d['refuse']} correct={d['correct']} "
              f"incorrect={d['incorrect']} | marker-agree={d['marker_vs_llm_agreement']} "
              f"| base_err_answered={d['base_err_answered']} "
              f"| majcorr/answered={d['n_majority_correct_answered']}/{d['n_answered_entities']}")


if __name__ == "__main__":
    main()
