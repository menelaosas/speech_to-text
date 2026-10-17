# Known issues

## Greek final sigma (ς) is dropped

**Symptom.** On conversational recordings the raw model output has words like `σα`, `όμω`, `μήνε`,
`τη`/`του` (for `της`/`τους`) instead of `σας`, `όμως`, `μήνες`…

**Cause.** A bug in the model itself, reported upstream in
[discussion #25](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/25) (unresolved).
We verified that:

- the tokenizer encodes and decodes `ς` correctly (token id 14855) and does not suppress it;
- the model ranks `ς` about 100th where it belongs, in both bf16 and fp32;
- on clean read speech it *does* write ς; on conversational audio it almost never does;
- audio filters, segment length and prompt changes don't help.

**Workaround (on by default for `-l el`).** After transcription, a language model (XLM-RoBERTa-large)
chooses between `w` and `w+ς` for every candidate word in context. This cuts ς-related word errors by
about 61%. Full details and benchmarks: [Accuracy](Accuracy.md).

```bash
--sigma-fix lm     # default, context-aware (needs models/xlm-roberta-large, ~2.2 GB)
--sigma-fix dict   # word-frequency list only: fast, no extra model, fixes fewer words
--sigma-fix off
```

About 17% of ς words are still missed, and some ς are added wrongly. Proofread important transcripts.

**If the model is updated upstream:** re-run `uv run speech-to-text download` and check with
`--sigma-fix off`. The adaptive margin already switches to cautious mode when the model writes ς normally.

## Repetition loops

**Symptom.** Occasionally the decoder gets stuck repeating a sentence, e.g. `Να κάτσουμε.` ×22.

**Workaround (always on).** `postprocess.collapse_repetitions` reduces any sentence repeated 4+
times in a row to 2 copies. Natural repeats such as "Ναι. Ναι. Ναι." are left alone.

## Hallucinated subtitle credits on very short clips

**Symptom.** On clips shorter than ~2.5 s (near-silence, a single word) Cohere sometimes writes
`Υπότιτλοι AUTHORWAVE` ("Subtitles AUTHORWAVE"), a credit memorised from subtitle training data.
Seen 5 times in 10 minutes when the pilot transcribed one clip per speaker turn.

**Workaround.** Don't feed the model very short clips: the root tool uses ~30 s segments and the pilot
uses VAD units ≤ 30 s. The pilot also flags such phrases (`known_hallucination_phrase`) without deleting
them. No occurrence was found in the current pilot output.

## Diarization: overlap and speaker count (pilot)

- pyannote marks **7.5× more overlapping speech** than the operator's reference (40.5 s vs 5.4 s in the
  excerpt), which makes up most of the 15.9% false alarm in DER. Either pyannote over-detects or the hand
  transcript under-marks overlap; it has not been decided yet.
- pyannote estimates **9 speakers** for the whole file; the operator heard 7 in the excerpt alone.
  If the number is known, set `input.num_speakers` in `pilot/config/pilot.yaml`.

## GPU non-determinism (pilot, fixed)

Word timestamps differed by one frame (20 ms) on a few words between runs, depending on which model had
used the GPU before in the same process. Fixed by running every GPU model in its own process
(`device.isolate_gpu_steps`). Two fresh runs are now identical.

## Other model limitations

- No automatic language detection.
- No word timestamps or speaker diarization in Cohere itself. The root tool's timestamps are per ~30 s
  segment; the pilot adds diarization and word timestamps with separate models.
- Capitalisation/punctuation may be inconsistent across segment boundaries.
- Proper names and military/technical terms are often misheard. There is no prompt or vocabulary-boosting
  option (the context slot was tested: the model copies it into the output).
