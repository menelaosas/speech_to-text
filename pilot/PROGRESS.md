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

## Reference scoring (operator transcript received)

- [x] Imported `manually transcripted.txt` → `reference/excerpt.txt` (source file only read; hash checked).
- [x] **WER 42.0% raw / 36.8% ς layer, CER 22.7% / 21.7%** on 571 words (spec normaliser, numbers to words).
      Variants with lenient normaliser and numbers as written in REPORT.md.
- [x] Measured the open choices on the excerpt (D-16): BF16 = FP32, beam 4 slightly better than greedy,
      hybrid kept, min_context unchanged. DER not computed (no speaker timings in the reference).
- [x] REPORT.md now has a *Measured accuracy* section and updated recommendations.

## Improvements 1 + 2 (operator request: word timestamps, DER)

- [x] **Greek CTC aligner** (own Viterbi; `jonatasgrosman/wav2vec2-large-xlsr-53-greek`, timing only). 10,985 of
      11,099 tokens aligned on the full file, 41 s on GPU, 1.6 GB peak.
- [x] **Aligned mode** (D-17, now default): long VAD units → words timed → split by speaker, speaker changes
      snapped to sentence ends. **cpWER 81.8% → 61.7%** at the same WER (37.0% ς layer).
- [x] **DER** (D-18): reference RTTM from the operator's labels + forced-alignment times. **DER 36.6%**
      (collar 0.25 s); 26.8% with exclusive diarization: pyannote marks 7.5× the reference's overlap.
- [x] Fixed: RTTM loader dropped overlapping reference turns; cache did not invalidate after an upstream
      recompute (D-20); 20 ms word-time non-determinism → GPU steps in isolated processes (D-19).
- [x] **Deliverable: `runs/pilot-final/`** (fresh, 321 s wall, all steps computed, identical to
      `pilot-final-repeat`, SHA unchanged, auto-scored against the operator reference). 26 tests pass.

**Stopped here as instructed.** Waiting for the operator: hand transcript of `reference/excerpt.wav`, speaker
count if known, and approval before anything beyond the pilot. Stretch steps A/B not started.

## Benchmarks (2026-10-17)

- [x] Common Voice 27.0 Greek moved into `data/eval/common_voice/` (29,467 clips, unchanged).
- [x] `speech-pilot benchmark` / `benchmark-compare` (D-21); `pyarrow` pinned; 38 tests pass.
- [x] Baselines: CV dev 8.92%, CV test 9.77%, FLEURS 7.42% WER; beam search confirmed significant.

