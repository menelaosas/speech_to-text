# Progress

## 2026-10-17

- [x] **Section 2 inputs.** Asked the operator: speakers unknown, phone on the table. File path, model
      path and language were known. pyannote not downloaded.
- [x] **Environment audit.** See `ENVIRONMENT_AUDIT.md`. SHA-256 baseline
      `8b4ff418…12004785`. Chain-of-custody note: the file had been moved with `mv` on 2026-10-16,
      before the pilot.
- [x] **Isolated environment.** `pilot/.venv` on Python 3.12.14, exact pins, host Python 3.14 untouched.
      Added librosa (required by the Cohere feature extractor). Cohere model reused, not re-downloaded.
- [x] **pyannote access.** Operator accepted the terms; downloaded revision `3533c8cf…` to
      `models/pyannote-speaker-diarization-community-1`. Pipeline runs offline (HF_HUB_OFFLINE=1).
- [x] **Smoke test, Cohere 30 s** (`runs/smoke-*/smoke.json`): cuda:0, load 3.9 s, process 0.86 s
      (RTF 0.029), peak 4043 MiB allocated / 4416 MiB reserved, VRAM back to baseline (246 → 356 MiB),
      SHA unchanged.
- [x] **Smoke test, diarization 2 min**: cuda, load 1.5 s, process 3.3 s (RTF 0.028), peak 858 MiB
      allocated / 1094 MiB reserved, VRAM back to baseline, 2 speakers, 29 turns.
- [x] **Pipeline code** (section 4): providers, pipeline, schema, gpu, export, qc, reference, score.
      13 unit tests pass.
- [x] **BF16 vs FP32** (D-09): 4/5 units identical, 1 differs in a garbled passage; FP32 can't fit 8 GB.
      BF16 kept, open question until the reference exists.
- [x] **Steps 1–3 on the 10-min excerpt** (`runs/excerpt-10min`): inspect (−23.3 LUFS, LRA 16.3 LU,
      true peak +0.1 dBFS, 5 near-full-scale samples of 226 M, no clipping runs), 16 kHz copy, Silero
      VAD on CPU: 523 s speech / 77 s non-speech, 113 regions, 5.3 s (RTF 0.009). Stops at
      diarization: pyannote missing.
- [x] **10-minute excerpt, segmentation comparison** (D-11): turns produced 5 hallucinations on very
      short units, VAD mixed speakers in 30 s blocks. Built **hybrid** + hallucination flag (D-12) +
      VAD gap filling (D-13). 15 tests pass.
- [x] **Full file** (`runs/full`, 252 s for 78.5 min, RTF 0.054): 440 segments, 9 speakers (estimated),
      231 overlap segments, 4 warning flags, SHA unchanged. Fixed after reviewing its report: hybrid wording,
      suspicious-segment ranking (overlap severity), exact uncovered-speech figure, honest wall time when
      steps are reused (D-14).
- [x] **Determinism**: `full` vs `full-repeat` identical (440/440); fresh `final` vs `full` identical.
- [x] **Final deliverable: `runs/final/`** (fresh run with the final code). Reference excerpt exported:
      `runs/final/reference/excerpt.wav`, 00:31:38–00:35:08, 4 speakers, 40.6 s overlap.

## Definition of done

- [x] Steps 1–9 on the full file, one command (`uv run speech-pilot run --run-id final`), no manual steps.
- [x] Cohere on GPU: device `cuda` recorded, peak 4351 MiB allocated (metrics.json).
- [x] Diarization ran; 9 labels used consistently across the file (one pass over the whole recording).
- [x] Overlaps are marked (`overlap`, all active speakers, `[OVERLAP]` in TXT/SRT), not hidden.
- [x] All section 6 outputs exist and parse (JSON schema, 440 SRT cues, RTTM loads in pyannote, WAV).
- [x] Original SHA-256 unchanged (`8b4ff418…12004785`, checked at the start and end of every run).
- [x] Second run with the same config gives the same transcript.
- [x] REPORT.md separates tested from not tested.
- [x] Reference excerpt exported (section 7).

**Stopped here as instructed.** Waiting for the operator: hand transcript of `reference/excerpt.wav`, speaker
count if known, and approval before anything beyond the pilot. Stretch steps A/B not started.
