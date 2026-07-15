import json
import re
import time
from pathlib import Path

import pandas as pd
import anthropic

from bielik_hallu import config
from bielik_hallu.dataset import candidates
from bielik_hallu.dataset.label import sample_answers, judge_known, label_row
from bielik_hallu.dataset.tokenization import tokenization_metadata
from bielik_hallu.extract.model import load_model


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in config.REFUSAL_MARKERS)


def parse_correct_verdict(text: str) -> bool:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(0))
        return bool(obj["correct"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def call_with_retry(fn, tries: int = 3, base_delay: float = 1.0, sleep=time.sleep):
    """Call fn(), retrying on Exception with exponential backoff.

    Re-raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad for retry
            last_exc = exc
            if attempt < tries - 1:
                sleep(base_delay * 2**attempt)
    raise last_exc


def claude_judge(entity: str, answer: str) -> bool:
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.JUDGE_MODEL,
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
    return parse_correct_verdict(msg.content[0].text)


def build_dataset() -> Path:
    model, tokenizer = load_model()
    checkpoint_out = config.DATA_DIR / "labeled.parquet"
    # Resume: reuse any entities already completed in a prior (possibly crashed)
    # run so a re-run doesn't repeat paid judge calls or lose progress.
    rows = []
    done: set = set()
    if checkpoint_out.exists():
        rows = pd.read_parquet(checkpoint_out).to_dict("records")
        done = {r["entity"] for r in rows}
    condition_lists = {
        "KNOWN": candidates.KNOWN,
        "UNKNOWN_REAL": candidates.UNKNOWN_REAL,
        "FABRICATED": candidates.FABRICATED,
    }
    for condition, entities in condition_lists.items():
        for entity in entities:
            if entity in done:
                continue
            answers = sample_answers(model, tokenizer, entity)
            all_refusal = all(is_refusal(a) for a in answers)
            all_correct = (
                judge_known(
                    entity, answers,
                    lambda e, a: call_with_retry(
                        lambda: claude_judge(e, a), tries=8, base_delay=2.0
                    ),
                )
                if condition == "KNOWN" else False
            )
            label = label_row(condition, all_correct)
            if condition in ("FABRICATED", "UNKNOWN_REAL") and all_refusal:
                label = 0  # refusal = no hallucination
            md = tokenization_metadata(tokenizer, entity)
            rows.append({
                "entity": entity,
                "condition": condition,
                "prompt": config.PROMPT_TEMPLATE.format(entity=entity),
                "answers": answers,
                "all_refusal": all_refusal,
                "all_correct": all_correct,
                "label_hallucination": label,
                **md,
            })

            # Incremental checkpoint: persist progress after each entity so a
            # mid-run crash doesn't lose completed work.
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_out = config.DATA_DIR / "labeled.parquet"
            pd.DataFrame(rows).to_parquet(checkpoint_out)

    df = pd.DataFrame(rows)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / "labeled.parquet"
    df.to_parquet(out)
    return out
