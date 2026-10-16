# Development

## Setup

```bash
uv sync            # includes dev dependencies (pytest)
uv run pytest      # unit tests (no GPU or model needed)
```

## Conventions

- `src/` layout, package `speech_to_text`. The entry point `speech-to-text` is defined in `pyproject.toml`.
- Dependencies are managed with `uv add <pkg>` / `uv add --dev <pkg>`. Commit `uv.lock`.
- Model weights, audio and transcripts are **never** committed (see `.gitignore`).
- Heavy imports (torch, transformers) happen lazily inside CLI commands so `--help` stays fast.

## Tests

| File | Covers |
|---|---|
| `tests/test_audio.py` | Segmenting: no gaps or overlaps, max length, cuts at silence |
| `tests/test_greek.py` | Final-sigma restoration |
| `tests/test_postprocess.py` | Repetition-loop collapsing |
| `tests/test_writers.py` | Timestamp formatting, txt/srt/json output |

Manual end-to-end check:

```bash
ffmpeg -i data/input/some.m4a -t 120 -c copy /tmp/sample.m4a
uv run speech-to-text transcribe /tmp/sample.m4a -o /tmp/out -f txt
```

## Ideas / TODO

- Optional voice-activity detection (e.g. Silero VAD) for better segment boundaries
- Speaker diarization (pyannote) merged with segments
- vLLM server backend for higher throughput
- Larger or custom Greek dictionary for the ς fix
