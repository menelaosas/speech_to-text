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
| --- | --- |
| `tests/test_audio.py` | Segmenting: no gaps or overlaps, max length, cuts at silence |
| `tests/test_greek.py` | Final-sigma restoration (the LM test is skipped if `models/xlm-roberta-large` is missing) |
| `tests/test_postprocess.py` | Repetition-loop collapsing |
| `tests/test_writers.py` | Timestamp formatting, txt/srt/json output |

Accuracy benchmarks: `scripts/evaluate.py`, see [Accuracy](Accuracy.md). Run them before and after
any change to decoding or post-processing.

Manual end-to-end check:

```bash
ffmpeg -i data/input/some.m4a -t 120 -c copy /tmp/sample.m4a
uv run speech-to-text transcribe /tmp/sample.m4a -o /tmp/out -f txt
```

## Pilot (`pilot/`)

Separate uv project, Python 3.12, exact pins. 38 tests (no models needed):

```bash
cd pilot && uv run pytest
```

| File | Covers |
| --- | --- |
| `tests/test_units_attribution.py` | overlap regions, turn/VAD/hybrid unit building, speaker attribution |
| `tests/test_alignment.py` | CTC Viterbi, Greek character mapping, word grouping, sentence snapping, cpWER, RTTM loading |
| `tests/test_qc_export.py` | quality flags, TXT/SRT/RTTM formats, schema round trip |
| `tests/test_greek_text.py` | scoring normaliser, Greek number verbaliser, reference import |
| `tests/test_cache.py` | an upstream change invalidates downstream steps |
| `tests/test_benchmark.py` | benchmark scoring (Greek normaliser), seeded subsets, Common Voice loading, paired comparison verdicts |
| `tests/test_web.py` | Speech Studio: save/reopen/export, audio ranges, uploads, host/origin checks, edit validation, import |

The web UI (`web.py` + `web_assets/`) is plain HTML/CSS/JS served by the Python standard library, with
no frontend build. It must never load models itself: it calls the pilot CLI as a subprocess
([Web UI](Web-UI.md)).

Conventions: no paths or hardware values in code (all in `config/pilot.yaml`); every model behind a
provider interface (`providers/`); every decision recorded with its evidence in `pilot/DECISIONS.md`.
Measure pipeline changes with `speech-pilot benchmark` + `benchmark-compare` (Common Voice dev, then test; see
[Benchmarks](Benchmarks.md)) and against the meeting reference with `scripts/compare_on_reference.py`.

## Ideas / TODO

- vLLM server backend for higher throughput
- Larger or custom Greek dictionary for the ς fix
