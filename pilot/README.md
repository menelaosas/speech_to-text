# Judicial Speech AI: pilot

Speaker-attributed Greek transcript of a single recording: inspection → 16 kHz copy → Silero VAD →
pyannote diarization → Cohere ASR → Greek ς layer → CTC word alignment → speaker attribution → quality
flags → JSON/TXT/SRT/RTTM + report, with a SHA-256 integrity check of the original. Use it from the
command line (`speech-pilot`) or in the browser (`speech-pilot-web`, see [Browser workspace](#browser-workspace)).
This is a pilot, not the platform.

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

## Browser workspace

From `pilot/`, start the app:

```bash
uv sync
uv run speech-pilot-web
# Or, using an already installed pilot environment:
.venv/bin/python -m speech_pilot.web
```

Open **http://127.0.0.1:8765** in your browser. No Node.js installation or frontend build is needed.
The server binds to this computer's loopback interface. Keep the terminal running while you work.

1. Choose or drop an audio file (up to 2 GiB), then click **Transcribe recording**. An upload progress
   bar shows the transfer.
2. Follow the 11 pipeline steps (the current one is marked) and the log. Jobs queue one at a time, using
   the isolated GPU pipeline. If a job fails, the failed step is marked and **Try again** re-runs it
   with the same upload (finished steps are reused); **Delete recording** removes it.
3. Play the recording. The player stays at the top while you scroll and shows who is speaking. The
   current word is highlighted; overlapping segments can highlight together. Each speaker has its own
   colour, and overlap segments list the other speakers ("with SPEAKER_02"). Click a word to seek, or a
   segment's ▶ timestamp to play from there. Uncheck **Follow playback** to scroll freely.
4. Double-click a word (or focus it and press Enter) to correct it. The dialog shows the word in context,
   what the machine wrote (**Restore** brings it back) and **Listen again** to replay just that word.
   Enter applies the correction. A correction can be a phrase, or blank to remove the token; it keeps the
   original word's time interval (no realignment). Dotted underlines mark estimated word timing.
5. **Speakers** panel: rename a speaker in every segment at once (e.g. SPEAKER_01 → a name). Single
   turns can still be changed in their own speaker field (suggestions list existing names).
6. Find text with the search box (matches are highlighted and counted). **Show** filters to *Needs
   review* (any flag, including overlap) or *Warnings only* (flags other than overlap alone);
   **Next to review** jumps to the next one.
7. **Save new version** (enabled when there are unsaved corrections), then use **Download**: the
   *machine transcript* (as produced, no corrections) or *your saved version* in TXT, SRT or JSON. JSON
   contains the original transcript and the separate corrections. Saved corrections and the latest
   version reopen when you select that recording again. Leaving with unsaved corrections asks first.

Keyboard: `Space` play/pause · `←`/`→` 5 s · `/` search · `Ctrl`/`⌘`+`S` save · `Enter` edit the
focused word, `←`/`→` move between words of a segment.

All configured model folders must already exist (see `config/pilot.yaml`); inference is offline.
In addition to Cohere, pyannote, and XLM-R, aligned mode requires:

```bash
uv run hf download jonatasgrosman/wav2vec2-large-xlsr-53-greek \
    --revision 489b34fb35fc5876af6193d419772cb9d6d1d531 \
    --local-dir ../models/wav2vec2-large-xlsr-53-greek
```

The app uses `config/pilot.yaml`, forcing `asr.segmentation=aligned`, processing the full uploaded file,
and clearing `reference.reuse_dir` so another recording's human reference is never reused. Other model,
language, decoding, and device settings come from the chosen config. The default requires CUDA.
Missing models or pipeline failures appear in the processing log; failed jobs can be uploaded again.
On server restart, interrupted jobs are marked failed; completed recordings and saved edits persist.

To review a completed run without running models again:

```bash
uv run speech-pilot-web --import-run runs/final-v2
```

This copies its transcript and prepared WAV into the workspace, leaving the original run untouched.
Import once; later starts can omit `--import-run`. Older runs without word timestamps support
segment highlighting and full-segment text editing. Upload and transcribe again for word highlighting.

Use `--port 9000`, `--config config/custom.yaml`, or `--data-dir /path/to/workspace` to override defaults.
Recordings live in ignored `pilot/.web-data/<job-id>/`: uploaded audio, `pipeline.log`, `run/` outputs,
`draft.json` (last saved corrections), and `edits/<version>.txt|srt|json`. Each explicit save creates new
files; canonical `run/transcript.json` is never overwritten. Uploads, WAV copies, and pipeline outputs
consume local disk space. Stop the server with Ctrl+C; an active pipeline is allowed to finish.

Backend/API checks (use the pilot environment):

```bash
uv run pytest tests/test_web.py -q
```

## Benchmarks (measure an improvement)

WER/CER on public Greek read speech, with the pipeline's audio preparation, Cohere settings and ς layer
(no speakers involved). Data: Common Voice 27.0 Greek in `../data/eval/common_voice/…/el/`, FLEURS in
`../data/eval/fleurs/`.

```bash
uv run speech-pilot benchmark --dataset cv --split dev --name baseline-cv-dev     # ~1 min
uv run speech-pilot benchmark --dataset cv --split dev --name try-x --set asr.num_beams=8
uv run speech-pilot benchmark-compare runs/benchmarks/baseline-cv-dev runs/benchmarks/try-x
```

Try changes on `dev`, confirm on `test`. `benchmark-compare` reports the difference with a 95% interval and
only says "better"/"worse" when it is significant. Baselines: CV dev **8.92%**, CV test **9.77%**, FLEURS
**7.42%** WER. Details: [Benchmarks wiki page](../docs/wiki/Benchmarks.md).

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

`ENVIRONMENT_AUDIT.md`, `PROGRESS.md`, `DECISIONS.md` (D-01 … D-20).

Wiki: [Pilot](../docs/wiki/Pilot.md) (overview and measured results), [Benchmarks](../docs/wiki/Benchmarks.md) and
[Web UI](../docs/wiki/Web-UI.md) (Speech Studio workflow, storage, security, troubleshooting).
