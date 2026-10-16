"""Project-wide defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MODEL_DIR = PROJECT_ROOT / "models" / "cohere-transcribe-03-2026"

# Masked language model used to restore the Greek final sigma (see greek.py).
SIGMA_MODEL_ID = "FacebookAI/xlm-roberta-large"
SIGMA_MODEL_DIR = PROJECT_ROOT / "models" / "xlm-roberta-large"

INPUT_DIR =PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

SAMPLE_RATE = 16_000

# The model has no automatic language detection, so a language must always be given.
DEFAULT_LANGUAGE = "el"
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "el": "Greek",
    "nl": "Dutch",
    "pl": "Polish",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "ar": "Arabic",
}

# The model handles clips up to 35 s; staying below that avoids its internal re-chunking.
MAX_SEGMENT_S = 30.0
# Window at the end of each segment in which we look for the quietest split point.
SPLIT_SEARCH_S = 5.0

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".webm", ".mp4", ".mkv"}
