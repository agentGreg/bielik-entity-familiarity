import os
from pathlib import Path

# Model is overridable via env (BIELIK_MODEL_ID) to support the scale sweep
# (1.5B / 4.5B / 11B) without editing this file.
MODEL_ID = os.environ.get("BIELIK_MODEL_ID", "speakleash/Bielik-1.5B-v3.0-Instruct")
MODEL_SLUG = MODEL_ID.split("/")[-1]
DEVICE = "mps"
DTYPE = "bfloat16"

CONDITIONS = ("KNOWN", "UNKNOWN_REAL", "FABRICATED")
# Domain string: prompt is sent to a Polish model, kept in Polish on purpose.
PROMPT_TEMPLATE = "Kim jest {entity}? Odpowiedz jednym zdaniem."
N_SAMPLES = 5

# Outputs are scoped per model so sweep runs do not overwrite each other.
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data" / MODEL_SLUG
RESULTS_DIR = _ROOT / "results" / MODEL_SLUG

JUDGE_MODEL = "claude-opus-4-8"
# Domain strings: markers of a Polish-language refusal from the model.
REFUSAL_MARKERS = ("nie wiem", "nie znam", "nie mam informacji", "brak informacji",
                   "nie jestem w stanie", "nie posiadam")
