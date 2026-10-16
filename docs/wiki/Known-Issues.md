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

## Other model limitations

- No automatic language detection.
- No word timestamps or speaker diarization. Timestamps are per ~30 s segment.
- Capitalisation/punctuation may be inconsistent across segment boundaries.
- Proper names and military/technical terms are often misheard. There is no prompt or vocabulary-boosting
  option (the context slot was tested: the model copies it into the output).
