# Decisions

Each entry: what was decided, why, and the evidence. Newest last.

## D-01 Separate `pilot/` project next to the existing tool (operator, 2026-10-17)
The pilot is its own uv project with exact pins (`pyproject.toml` + `uv.lock`). It reuses the proven
parts of `speech_to_text`: segmentation at quiet points, beam search, Greek ς restoration and the
repetition detector. The old CLI keeps working.

## D-02 Canonical text = raw Cohere output; ς restoration is a separate layer (operator, 2026-10-17)
The rule "never correct or invent text" applies to the canonical `text`. Greek final-ς restoration
(XLM-RoBERTa-large) is stored in `text_normalized` and `transcript.normalized.txt`, labelled as
machine-normalised, with its model and revision in the provenance. Repetition loops are **flagged**,
not collapsed. TXT/SRT show raw text.
Evidence for keeping the layer: on Greek FLEURS with all ς removed, word accuracy 82.1% → 95.6%
(speech_to_text `docs/wiki/Accuracy.md`). This recording has a raw ς rate of ~0%, so the raw text
lacks ς throughout (upstream bug, model discussion #25).

## D-03 Silero VAD runs on CPU
Silero is a ~2 MB model designed for CPU inference. Running it on CPU keeps the GPU free for the
heavy models ("one heavy model at a time"). This is a deliberate choice, recorded per step in
`metrics.json` (device: cpu), not a fallback.

## D-04 Beam search with 4 beams, no sampling
Deterministic decoding (`do_sample=False`). Measured on Greek FLEURS (200 clips): WER 9.15% (greedy) →
8.86% (4 beams), and it removed a hallucinated opening sentence on this recording. Costs ~1.8× ASR time.

## D-05 ASR units ≤ 30 s, cut at quiet points
Cohere's limit is 35 s, and the processor re-chunks longer input internally (losing our boundaries).
Units are kept ≤ 30 s by cutting at the lowest-energy 100 ms window in the last 5 s. The ASR provider
raises an error if the processor ever re-chunks a unit.

## D-06 Overlap handling
ASR units come from the exclusive diarization. When two turns of different speakers meet inside an
overlap region of the overlap-aware diarization, they are merged into one unit, transcribed once and
marked `overlap: true` with all active speakers. This avoids false sequential turns. Speakers are
listed when they cover ≥ 0.5 s or ≥ 15% of the segment.

## D-07 Pinned torch 2.14.1 with torchaudio 2.11.0
uv resolved pyannote.audio 4.0.7 with torch 2.14.1+cu130 and torchaudio 2.11.0+cu130 (latest
torchaudio). The version mismatch was checked: torchaudio resampling, torchcodec decoding, pyannote and
Silero all import and run. Cohere ran on cuda:0 in the smoke test.

## D-08 Greek scoring normaliser
`greek_text.normalize`: lowercase; strip tonos and dialytika; ς→σ; numbers to words (own verbaliser,
because num2words has no Greek; neuter forms, feminine for thousands); strip punctuation; unify
εφτα/οχτω/εννια. WER/CER are reported with and without number normalisation, for both the raw and the
ς-normalised layer. Known limit: gendered numerals ("τρεις"/"τρία") in the reference are not unified.

## D-09 BF16 on GPU for Cohere (open question noted)
Comparison on the same 2-minute excerpt (01:00–03:00), same 5 VAD units, same decoding (4 beams):
BF16 on cuda vs FP32 on CPU. 4 of 5 units identical. One unit differs in a garbled passage
(word difference 5.5%, character difference 3.6% over the 2 minutes). Neither output can be called
more accurate without a reference.
FP32 does not fit: 2B parameters × 4 bytes ≈ 8.3 GB > 8 GB VRAM. FP32 on CPU took 105 s for 2 min
(BF16 GPU: 5.6 s incl. load). The definition of done requires Cohere on the GPU, so the pilot uses BF16,
the dtype the weights are published in.
**Open:** once `reference/excerpt.txt` exists, score BF16 against FP32 on that excerpt to settle it.
Raw data: `runs/smoke-20261017-001245/precision_compare.json`.

## D-10 Step reuse is keyed on configuration, not code
A step is reused when its config sections are unchanged. After a code change, re-run the affected step
with `--force <step>` (or `--force all`). Final runs use fresh run ids.

## D-11 ASR segmentation: hybrid (diarization turns + minimum context + VAD gap filling)
Tried all three on the 10-minute excerpt (00:00–10:00), same models and decoding:

| Mode | Segments | Known hallucinations | text_in_nonspeech | Multi-speaker segments | VAD speech not transcribed |
| --- | --- | --- | --- | --- | --- |
| turns (one unit per turn) | 89 | **5** ("Υπότιτλοι AUTHORWAVE") | 4 | 17 | 28.6 s |
| vad (regions merged to 30 s) | 22 | 0 | 0 | 17 of 22 (blocks mix up to 3 speakers) | 0 s |
| hybrid | 63 | 0 | 0 | ~20 of 63 | 9.3 s (slivers < 0.5 s) |

- *turns*: 25 of 89 units were < 2 s. All 5 hallucinations were on units < 2.5 s (memorised subtitle
  credits, a known ASR failure on near-silence/very short clips).
- *vad*: most coherent text, no hallucinations, but 30-second blocks cannot be attributed to one
  speaker. This defeats the purpose of diarization.
- *hybrid*: start from turns, add VAD speech that no diarization turn covers (31.8 s on the excerpt,
  e.g. 00:09–00:22) as speaker-less units, then merge every unit < 3 s into its nearest neighbour
  (gap ≤ 1 s, result ≤ 30 s). Keeps turn-level attribution, removes the hallucinations, and drops no
  speech silently. Its text is closer to the VAD output than the turns output (WER between outputs 0.19
  vs 0.21). That is a proxy, **not** an accuracy measurement.
Chosen: **hybrid** (config `asr.segmentation`). Data: `runs/excerpt-10min-{turns,vad,hybrid}`.

## D-12 Known hallucination phrases are flagged, not removed
`qc.hallucination_phrases` (config) lists memorised strings ("Υπότιτλοι", "AUTHORWAVE", …). A match adds
the warning flag `known_hallucination_phrase` (highest weight in the suspicious-segment ranking). The text
is left as the model produced it.

## D-13 Speech without a diarized speaker is transcribed and flagged
VAD speech with no diarization turn gets primary speaker `null`, is shown as `UNKNOWN?` and is flagged
`no_diarized_speaker`. The amount of VAD speech still not inside any ASR unit is reported
(`vad_speech_not_transcribed_s`).

## D-14 Honest timings when steps are reused
Re-running a run id reuses cached steps, so that invocation's wall time is not the processing time.
`metrics.json` records `reused_steps` and `steps_compute_s`, and the report says so explicitly. The
deliverable `runs/final` is a fresh run (no reused steps). `speech-pilot report --run <dir>` rebuilds
REPORT.md from saved files without touching metrics (used after `compare` writes `determinism.json`).

## D-15 Review ranking
The "10 most suspicious segments" are ranked by flag weights (hallucination phrase 4, repetition /
non-speech text / empty / truncation 3, words-per-second 2, no diarized speaker 1.5, uncertain speaker 1),
plus, for overlap segments, 2 × the overlapped share of the segment + 0.5 per speaker beyond two.
Ranking only orders the review list; it never changes the transcript.
