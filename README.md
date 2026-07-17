# Graded Entity-Familiarity Readouts in Language Models

**Polish adaptation, cross-language robustness, and refusal steering.**

Author: **Grzegorz Brzezinka** (Prosit AS)
Contact: greg@prosit.no
Paper: preprint link to be added. A condensed version is in preparation for peer review
(venue withheld). Feedback welcome.

Companion to [*Does Bielik Know What It Doesn't Know?*](https://github.com/agentGreg/bielik-hallucination-detection),
which established that activation dispersion at the prompt point separates known
from fabricated Polish entities. That result was binary, single-family,
single-language, and correlational. This work asks what the signal **is**.

---

A language model asked about an entity it has never seen can abstain or
confabulate. We read the residual stream at the **final prompt token** — before
any answer token exists — in twelve instruction-tuned models from the **Bielik,
PLLuM, Gemma-4, and Qwen3** families, plus two base checkpoints as controls,
using a new popularity-graded dataset of 1,440 Polish entities (`data/`).

## TL;DR findings

- **Familiarity is a graded quantity, not a bit.** A familiarity probe separates
  real from fabricated entities in every family (AUROC 0.86–0.93 on the 320-entity
  eval subset). In the Polish-adapted **Bielik** and **PLLuM** families its score
  additionally tracks entity popularity (model-mean Spearman ρ 0.28–0.57, up to
  0.567 for Bielik-11B), versus at most 0.11 in Gemma and Qwen.
- **Gradation tracks Polish adaptation, not scale.** Two controlled before/after
  comparisons at matched architecture: the **Mistral-NeMo-12B** base grades at
  ρ=0.10 versus PLLuM-12B's 0.28, and the **Llama-3.1-8B** base at 0.19 versus
  Llama-PLLuM-8B's 0.36. The bases sit with the other non-Polish models; Polish
  continual pretraining is what moves them.
- **The readout is robust to the question language.** With the Polish question
  stem swapped for English around unchanged entity names, probes retain 96–101%
  of within-language AUROC. On **entity-disjoint** splits (train and test on
  different entities) transfer holds at 98–100% for Bielik but weakens to 74–93%
  for the multilingual Gemma.
- **The model reads this signal.** In Gemma-4-12B, the only model that natively
  refuses, adding a one-dimensional familiarity direction at a single layer moves
  refusal rates monotonically in both directions (0.24→1.00 on well-known
  entities; 0.73→0.00 on unknown ones).
- **Familiarity ≠ correctness, and refusals must be separated from both.** The
  familiarity signal tracks entity status, not answer accuracy. When a strict
  judge scores explicit refusals as "correct", every behavioral number is
  confounded for the one abstaining model; separating answered-from-abstained
  shows Gemma's base error among *answered* items is 0.90, not the refusal-
  inflated 0.60. A three-way LLM judge (refuse/correct/incorrect) agrees with
  the refusal-marker separation at 0.99.
- **Not a lexical shortcut.** Purely lexical classifiers on the entity string top
  out at AUROC 0.786 (character n-grams), well below the probe.

## What's here

```
data/            the 1,440-entity Polish dataset as CSV (+ data dictionary)
src/bielik_hallu/ core package: dataset build, extraction, metrics, analysis, risk
scripts/         one script per experiment (RQ1–RQ4 + robustness experiments)
```

Raw extraction artifacts (per-model hidden states, `signals.parquet`,
generations, judge verdicts) are **not** shipped — they are large and fully
regenerable from the code and dataset here.

## Reproduce

```bash
uv sync                        # or: pip install -e .
export ANTHROPIC_API_KEY=...   # only for the answer-correctness judge (RQ4)

# 1. Build the dataset (already provided in data/; this regenerates it)
uv run python scripts/build_dataset_v2.py

# 2. Extract prompt-point signals for a model (repeat per model)
uv run python scripts/run_v2_campaign.py --models Bielik-11B

# 3. RQ2 — familiarity gradation vs popularity
uv run python scripts/analyze_v2.py

# 4. RQ4 — pre-generation gating vs post-generation detectors
uv run python scripts/p2_generate_eval_subset.py
uv run python scripts/p2_calibration.py
uv run python scripts/p2_rq4.py

# 5. Figures
uv run python scripts/paper2_figures.py
```

RQ1 (cross-language transfer, `analyze_language_transfer.py`) and RQ3 (refusal
steering, `analyze_steering.py`) use the predecessor's three-condition dataset;
the additional `p2_*` scripts are the robustness experiments reported in the
paper (entity-disjoint transfer, lexical baselines, incremental validity,
held-out steering direction, refusal separation, three-way judge).

## Models

Gated Bielik/PLLuM checkpoints require access on Hugging Face. The pipeline reads
`BIELIK_MODEL_ID` / the model list in `scripts/run_v2_campaign.py`; families
covered are Bielik (1.5B–11B), PLLuM (4B–12B), Gemma-4 (E4B, 12B), Qwen3
(1.7B–14B), plus the Llama-3.1-8B and Mistral-NeMo-12B base controls.

## Citation

```bibtex
@article{brzezinka2026graded,
  title  = {Graded Entity-Familiarity Readouts in Language Models: Polish
            Adaptation, Cross-Language Robustness, and Refusal Steering},
  author = {Brzezinka, Grzegorz},
  journal= {arXiv preprint arXiv:2607.13568},
  year   = {2026}
}
```

## License

Dataset provenance and licence: see [`data/README.md`](data/README.md). Code is released under the [MIT License](LICENSE). Copyright (c) 2026 Grzegorz Brzezinka
(Prosit AS).

## Acknowledgments

Built on the **Bielik** Polish LLM family from
[SpeakLeash](https://speakleash.org/) and the Bielik team. The Bielik v3.0
models used here are **gated** on Hugging Face — request access on each model
page before running the tools, same for PLLuM. Anthropic and OpenAI models were used as judges for answers in tests and as a coding assitant.
