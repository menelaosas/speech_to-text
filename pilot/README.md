# Judicial Speech AI: pilot

Speaker-attributed Greek transcript of a single recording: inspection → 16 kHz copy → Silero VAD →
pyannote diarization → Cohere ASR → speaker attribution → quality flags → JSON/TXT/SRT/RTTM + report,
with a SHA-256 integrity check of the original. This is a pilot, not the platform.

## Setup

```bash
cd pilot
uv sync                               # Python 3.12, exact pins
uv run hf auth login                  # once
# accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1, then:
uv run hf download pyannote/speaker-diarization-community-1 \
    --local-dir ../models/pyannote-speaker-diarization-community-1
```

Cohere Transcribe and XLM-RoBERTa-large are expected in `../models/` (see `config/pilot.yaml`).

## Run

```bash
uv run speech-pilot smoke                           # Cohere 30 s + diarization 2 min
uv run speech-pilot run --excerpt 0:600 --run-id excerpt-10min
uv run speech-pilot run --run-id full               # the whole file: steps 1-9 (= scripts/run_pilot.py)
uv run speech-pilot compare runs/full runs/full-repeat   # determinism check
uv run speech-pilot report --run runs/full          # rebuild REPORT.md (e.g. after compare)
uv run speech-pilot run --run-id pilot-final         # current deliverable (aligned mode, auto-scored)
uv run speech-pilot score --run runs/final --numbered "runs/final/reference/manually transcripted.txt"
uv run speech-pilot align-reference --run runs/final # time the operator's lines -> excerpt.rttm (DER)
uv run speech-pilot score --run runs/X --reference runs/final/reference   # score any run on the reference
uv run python scripts/compare_on_reference.py --run runs/final   # measure pipeline choices on the reference
```

`--set key=value` overrides config values (e.g. `--set asr.segmentation=vad`). Re-running with the same
`--run-id` reuses finished steps whose config is unchanged. `--force 05_asr` re-runs one step.

## Outputs (`runs/<run_id>/`)

| File | Content |
| --- | --- |
| `transcript.json` | canonical transcript: segments (with word timestamps), speakers, overlap, flags, provenance, models, config, SHA-256 |
| `transcript.txt` | `[00:12:31 – 00:12:44] SPEAKER_02: ...` (raw text) |
| `transcript.normalized.txt` | same with Greek final ς restored (machine-normalised, not canonical) |
| `transcript.srt` | subtitles with speaker labels |
| `diarization.rttm` | overlap-aware diarization |
| `metrics.json` | per-step device, timings, VRAM, RTF, GPU/power state |
| `REPORT.md` | summary, flags, 10 most suspicious segments, tested vs not tested |
| `reference/` | operator's reference excerpt (reused by every run) + `score.json` (WER/CER, cpWER, DER) |
| `01_inspect/` … `09_integrity/` | raw outputs of every step (rebuild without re-running models) |

## Documents

`ENVIRONMENT_AUDIT.md`, `PROGRESS.md`, `DECISIONS.md`.
