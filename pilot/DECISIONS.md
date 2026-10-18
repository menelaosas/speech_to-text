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

## D-16 Results against the operator's reference (2026-10-17)
Reference: `runs/final/reference/manually transcripted.txt` (operator, `<speaker number> <text>` lines),
imported unchanged in content to `excerpt.txt` (`excerpt_import.json` records the rule: multi-digit labels
such as `12` = speakers 1 and 2 at once). Excerpt 00:31:38–00:35:08, 571 reference words, 7 speakers.
DER not computed (no speaker timings).

| Measured | Result | Decision |
| --- | --- | --- |
| Raw Cohere text (runs/final) | WER 42.0%, CER 22.7% | canonical, unchanged |
| ς-normalised layer | WER 36.8% (−5.2 points, 30 fewer substitutions) | keep the layer (D-02 confirmed) |
| Beam 4 vs greedy (same units) | 39.1% vs 39.6% | keep beam 4 (D-04 confirmed, small gain) |
| BF16 GPU vs FP32 CPU (same 18 units) | 44.1% vs 44.3% raw, 14/18 identical | **BF16 confirmed, D-09 closed** |
| Segmentation on the excerpt | vad 35.9%, full-run hybrid 36.8%, turns 38.9% | keep hybrid: vad is ~1 point better but its 30 s blocks mix speakers |
| hybrid min_context 3/10/15 s (full-file diarization) | 36.8% / 37.5% / 36.4% | keep 3 s: differences are ~±6 words, within noise, and longer units lose attribution |

Not tuned on the test set: no config value was changed on the basis of this excerpt. The reference has
about 22 typing slips (30 of 561 words are not in a Greek word list, some are real rare words), so the WER
includes some reference noise. Data: `runs/final/reference/{score,comparison,context_comparison}.json`,
reproducible with `scripts/compare_on_reference.py`.

## D-17 Word timestamps: transcribe long units, split words by speaker ("aligned" mode, now default)
Long units have more context (fewer lost words) but mix speakers; short turn units attribute well but
lose words. Aligned mode gets both: VAD units ≤ 30 s go through Cohere; the raw words are timed by
Greek CTC forced alignment (`jonatasgrosman/wav2vec2-large-xlsr-53-greek` rev `489b34fb`, Apache-2.0,
used **only for timing**, never for text); each word goes to the diarization speaker most active during
it; consecutive words of one speaker form a segment; a speaker change is moved to the nearest sentence
end within 3 words (ASR punctuation), so word-timing jitter does not cut sentences.
Own Viterbi (torchaudio.forced_align was removed in 2.9); digits verbalised before alignment
(the acoustic model has no digit tokens); Latin-letter tokens are interpolated and marked `aligned: false`.
Every segment carries its words with start/end/score/speaker (`segments[].words`).

Measured on the reference excerpt, same full-file diarization for all rows:

| Variant | WER (ς layer) | cpWER (words + speaker) | Segments starting mid-sentence (whole file) |
| --- | --- | --- | --- |
| hybrid (previous default) | 36.8% | 81.8% | 6% |
| vad (30 s blocks) | 37.0% | 87.0% | – |
| aligned, no sentence snapping | 37.0% | 63.6% | 41% |
| **aligned + snapping (default)** | **37.0%** | **61.7%** | **14%** |

cpWER is the standard speaker-attributed metric: the text of each speaker is concatenated on both sides and
speakers are matched one-to-one. The 20-point gain (115 fewer errors of 571) is far above the noise level
(~6 words). Alignment of the whole file: 10,985 of 11,099 tokens aligned, 0 units failed, 42 s on GPU,
1.6 GB peak.

## D-18 DER reference from the operator's labels + forced-alignment timing
The operator gave speaker labels per line but no times. `speech-pilot align-reference` aligns the
operator's own words to `excerpt.wav` and takes each line's first/last word as its span →
`reference/excerpt.rttm` (marked derived in `excerpt_alignment.json`). Speaker identity is human; only
timing is machine. Caveats: 28 of 69 lines (33.5 s of 208 s, mostly short interjections and overlap)
aligned with low confidence; the operator marked 4 overlap lines (5.4 s).
Result for the full-file diarization (collar 0.25 s, overlap scored): **DER 36.6%** = missed 3.2% +
false alarm 15.9% + confusion 17.6%. With pyannote's *exclusive* output: **DER 26.8%** (false alarm 1.4%).
pyannote marked 40.5 s of overlap in the excerpt vs 5.4 s in the reference. Either pyannote over-detects
overlap or the operator under-marked it (common in hand transcripts). Overlap thresholds were not changed
on one excerpt (see recommendations). Fixed along the way: the RTTM loader dropped a speaker when two had
the same span (4 overlap turns lost); now one track per line, with a test.

## D-19 Every GPU step runs in its own process (determinism with word timestamps)
With word timestamps, two runs gave the same text, speakers and segments, but 5 of 11,099 word times
differed by one alignment frame (20 ms). Cause: the CTC aligner's GPU kernels behaved differently
depending on what had run before in the same process (aligner first vs after Cohere + XLM-R). Forcing
deterministic kernels (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` set before CUDA
starts) made repeated runs of the *same* path identical, but not across paths.
Fix (the spec's own fallback): `device.isolate_gpu_steps: true` runs diarization, ASR, ς layer and alignment
each in a fresh spawned process, in order. Verified identical, word times included: two fresh runs
(iso-a = iso-b), a fresh run vs a run that recomputed only the alignment (iso-a = iso-c), and vs final-v3.
Side effect: VRAM is fully released between models by process exit.

## D-20 Cache keys include upstream outputs
Found while testing D-19: a recomputed upstream step (e.g. `--force 05c_align`) did not invalidate later
steps, because reuse was keyed on configuration only (D-10). Now each step stores a hash of its output
files, and a step's cache key includes the output hashes of all earlier steps (test: `tests/test_cache.py`).
Runs made before this change are recomputed in full when reused.

## D-21 Benchmarks on public Greek read speech (2026-10-17)
The meeting reference (571 words) cannot show differences below ~3–4 points. The operator supplied
Common Voice 27.0 Greek, moved unchanged into `data/eval/common_voice/cv-corpus-27.0-2026-09-11/`.
`speech-pilot benchmark` runs the pipeline's audio preparation, ASR and ς layer on a split, and
`benchmark-compare` pairs two runs clip by clip with a bootstrap 95% interval (verdict only when it
excludes zero). Rule: iterate on `dev`, confirm on `test`.

| Run | WER raw (95% CI) | WER ς layer | CER raw |
| --- | --- | --- | --- |
| CV dev, 1,841 clips | 8.92% (8.24–9.66) | 9.14% | 2.45% |
| CV test, 1,842 clips | 9.77% (9.06–10.51) | 10.09% | 2.74% |
| FLEURS test, 650 clips | 7.42% (6.84–8.00) | 7.41% | 2.51% |
| CV dev, greedy | 9.18% | 9.40% | 2.54% |

- D-04 confirmed with enough data: 4 beams vs greedy −0.26 points, 95% CI [−0.46, −0.08].
- On read speech the ς layer is neutral to slightly worse (+0.2–0.3 points on CV); its gain is specific
  to audio where Cohere drops ς (meeting: −5.0 points). Candidate change, not adopted yet: skip the layer
  when the raw ς rate is normal.
- Caveats: read speech, not meetings; Cohere's training data is undisclosed, so absolute CV scores may be
  optimistic. Comparisons between variants remain valid.

## D-22 Second-model comparison and decoding safeguards (2026-10-18)

Operator selected comparison with a second model. Downloaded NVIDIA
`nvidia/parakeet-tdt-0.6b-v3` (CC BY 4.0) into ignored `models/`; its native Transformers provider runs
locally in FP32, batch size 1. Cohere keeps BF16, four beams, configured batches. Models are loaded
sequentially on the 8 GB GPU. No model weights were trained, no production recognizer was replaced,
and no original recording, reference, or saved transcript was edited.

### Same meeting excerpt, matching input units

`scripts/compare_asr_decoding.py` clips saved ASR unit boundaries to the 571-word human-reference
window. All variants receive the same nine audio units; reference text is used only after recognition.
This differs from extracting words from a full-file run, so these scores must not be compared as
before/after improvements against the previous 37.0% full-run result.

| Variant | WER raw | CER raw | WER final-sigma layer | CER final-sigma layer |
| --- | --- | --- | --- | --- |
| Cohere, best beam | 42.7% | 22.7% | 37.6% | 21.7% |
| Cohere, experimental four-candidate consensus | 42.2% | 22.4% | 37.1% | 21.4% |
| Parakeet TDT v3 | 49.2% | 29.1% | 45.2% | 28.4% |

Artifacts: `runs/decoding-20261017-204055/comparison.json` and `alternatives.md`.
This is a previously inspected development excerpt, not a held-out test. Consensus changes only a
few errors; the gain is too small to justify adoption from this excerpt alone. Parakeet is worse here.

### Matching 100-clip Common Voice dev sample

Same seeded sample, 652 reference words, independent of the meeting excerpt:

| Model | WER raw | CER raw | WER final-sigma layer | CER final-sigma layer |
| --- | --- | --- | --- | --- |
| Cohere | 9.82% | 2.66% | 9.97% | 2.69% |
| Parakeet | 18.87% | 4.45% | 10.28% | 2.96% |

Paired raw WER difference (Parakeet minus Cohere): +9.05 percentage points, bootstrap 95% interval
[+4.86, +13.49]. After the same sigma layer, the difference is +0.31 points, interval [-3.18, +3.55]:
no significant difference on this sample. Both raw and normalized scores are retained; the sigma
layer removes much of Parakeet's raw error here. This is a small read-speech dev sample, not evidence
that either model generalizes better to all Greek meetings; public-data training overlap is unknown.

Artifacts: `runs/benchmarks/cv-dev-n100-{cohere,parakeet}-20261018-002823/` and
`runs/benchmarks/cohere-parakeet-20261018-002823.json`.

Decision: keep Cohere as the production recognizer and consensus opt-in. Keep Parakeet as an explicit
comparison option in `benchmark --parakeet-model ...` and the reference comparison script.

### Correct stop detection and targeted recovery

Cohere now excludes EOS and padding from token counts and records the actual generation stop
condition. Before this change, 140 of the 191 full-file ASR rows included padding in `n_tokens`.
The examined run had no units at the 256-token limit, so token-limit recovery cannot explain or fix
its existing word errors. When a future unit really exhausts its budget without EOS, it can retry
once at up to 512 tokens, within decoder capacity, with one unit per retry. OOM or an empty retry
retains the first pass. Both attempts and beam candidates are kept in `asr_raw.json`; QC uses the
final stop condition, including for segments sharing a recovered ASR unit. Old runs without these
fields keep the legacy QC fallback. The new ASR config fields invalidate downstream caches normally.

## D-23 Review of the D-22 changes: kept; consensus stays off (2026-10-18)
The D-22 changes were reviewed and measured on the full Common Voice dev split (1,841 clips) and the full
78-minute recording before committing.

**No regressions with the defaults** (`num_candidates: 1`, `candidate_selection: best`):

- Common Voice dev, new code vs `baseline-cv-dev` (old code): identical on all 1,841 clips (WER 8.92%).
- Full pipeline re-run with diarization reused from `pilot-final`: identical, 671 of 671 segments.
- 47 tests pass.

**Kept, enabled:** token counts without EOS/padding (all 191 units previously included padding; the
largest real count is 202 of the 256 budget); retry of truncated generation (0 of 191 units were
truncated, so it never ran on this recording); QC on the real stop condition; config validation.
None of these changes WER/CER; they make counts and flags correct and add a safety net.

**Kept, comparison only:** Parakeet TDT v3 (worse than Cohere in D-22).

**Consensus selection: kept in the code, off by default.** On the full dev split it is significantly worse:

| Selection | WER raw (95% CI) | CER raw | Deletions |
| --- | --- | --- | --- |
| best beam (default) | 8.92% (8.24–9.66) | 2.45% | 105 |
| consensus, 4 candidates | 9.50% (8.80–10.23) | 2.58% | 153 |

Paired difference +0.57 points WER, 95% CI [+0.31, +0.86]; better on 34 clips, worse on 71. The small
gain seen on 100 clips and on the meeting excerpt (D-22) did not hold. Cause: of 124 clips where the choice
changed, 67 picked a shorter candidate and 10 a longer one, so deletions rose. Two reasons in
`select_candidate`: (1) edit distance is divided by the *other* candidate's length, so long candidates
receive larger penalties; (2) every distinct candidate has an equal vote, ignoring the model's own
beam scores. It stays available for experiments (`--set asr.num_candidates=4 --set
asr.candidate_selection=consensus`), and may become a default only if a corrected version (symmetric
normalisation, score-weighted votes) beats 8.92% on dev and is confirmed on test.
Data: `runs/benchmarks/review-{default,consensus}-cv-dev/`, `runs/review-pipeline/`.
