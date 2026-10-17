# Changelog

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
