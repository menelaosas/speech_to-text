# Changelog

## Benchmarks on Common Voice 27 Greek (2026-10-17)

- Common Voice 27.0 Greek moved into `data/eval/common_voice/` (29,467 clips; dev 1,841 / test 1,842)
- `speech-pilot benchmark`: WER/CER with 95% bootstrap intervals on Common Voice or FLEURS, using the
  pipeline's audio preparation, ASR settings and ς layer; deterministic subsets; full provenance
- `speech-pilot benchmark-compare`: paired clip-by-clip comparison with a verdict only when significant
- Baselines: CV dev 8.92%, CV test 9.77%, FLEURS 7.42% WER; beam 4 vs greedy confirmed (−0.26 points,
  significant); ς layer slightly worse on read speech (candidate improvement)
- `pyarrow` added to the pilot environment; 4 new tests (38). New wiki page: [Benchmarks](Benchmarks.md)

## Speech Studio UX update (2026-10-17)

- Fixed: **Enter** in the correction dialog cancelled the correction (now applies it); applying a
  correction re-rendered all ~11,000 words and **jumped the page to the top** (now updates one word in
  place); header counted "no speaker" as a speaker (10 → 9); "Verify" step never showed as done;
  duration shown as 78:27 (now 1:18:27); favicon 404s; connection-lost message never cleared
- Player pinned at the top while scrolling, with the current speaker; colour per speaker; overlap
  segments list the other speakers
- **Speakers** panel: rename a speaker in every segment at once; speaker fields suggest existing names
- Correction dialog: word marked in context, machine original with **Restore**, **Listen again**
- Review filter: *Needs review* (414) or *Warnings only* (89) on the test recording; **Next to review**
- Search highlights and counts matches; keyboard: `←`/`→` 5 s, `/` search, `Ctrl+S` save
- Download menu: machine transcript any time, latest saved version (also after reload)
- Processing: progress bar, current step, elapsed time; failed jobs show the failed step with
  **Try again** (reuses finished steps) and **Delete**; upload progress bar
- Higher contrast and larger small text; dismissible notices; confirmation dialogs instead of browser
  pop-ups; 2 new server tests (34 pilot tests)

## Speech Studio web UI (2026-10-17)

- `speech-pilot-web`: local browser app (127.0.0.1:8765) on top of the pilot. Upload, queued
  transcription with live step progress and log, playback with word highlighting, word and speaker-label
  corrections, saved versions, TXT/SRT/JSON export (JSON keeps original and corrections apart)
- Runs the unchanged pilot CLI as a subprocess with aligned mode forced and no reference reuse
- Loopback only, Host/Origin checks, strict CSP; corrections cannot change timestamps or word counts
- `--import-run` to review an existing pilot run without re-running models
- 6 web tests (32 pilot tests in total). New wiki page: [Web UI](Web-UI.md)

## Pilot 0.1 (2026-10-17): Judicial Speech AI pilot (`pilot/`)

- New pinned project: Silero VAD, pyannote community-1 diarization, Cohere ASR, attribution, quality flags,
  JSON/TXT/SRT/RTTM export, report, SHA-256 integrity check, one command for steps 1–9
- Raw Cohere text canonical; ς restoration kept as a separate layer; repetition loops and known
  hallucination phrases flagged, never removed
- Scored against the operator's hand transcript: WER 42.0% raw / 37.0% ς layer
- Word timestamps by Greek CTC forced alignment, words split by speaker: cpWER 81.8% → 61.7%
- DER 36.6% against the operator's labels with aligned times; pyannote over-marks overlap relative to it
- Every GPU step in its own process (deterministic word times); cache invalidates downstream steps
- New wiki page: [Pilot](Pilot.md); Accuracy, Known Issues, Installation, Troubleshooting, Development updated

## 0.2.0 (2026-10-16): accuracy

- Beam search (4 beams) by default: `--beams`
- Context-aware Greek ς restoration with XLM-RoBERTa-large and an adaptive margin: `--sigma-fix lm|dict|off`.
  ς-related word errors −61% when the model drops ς. FLEURS WER 9.15% → 8.78%.
- `download` also fetches XLM-R-large (`--skip-sigma-model` to skip)
- `scripts/evaluate.py`: WER/CER and ς-restoration benchmarks on Greek FLEURS
- Rejected after measurement: ς logit bias, prompt context, `<|itn|>`/emotion tokens, audio filters
- New wiki page: [Accuracy](Accuracy.md)

## 0.1.0 (2026-10-16)

- Initial project: uv package, `speech-to-text` CLI with `download` and `transcribe`
- Cohere Transcribe 03-2026 via native transformers support, bf16 on CUDA
- ffmpeg decoding, splitting into ≤ 30 s segments at quiet points, batched inference, silent segments skipped
- Output formats: txt, srt, json
- Greek final-sigma (ς) restoration workaround for upstream model bug
- Collapse decoder repetition loops
- Unit tests and wiki
