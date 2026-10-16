# Known issues

## Greek final sigma (ς) is never produced

**Symptom.** Greek transcripts have words like `σα`, `όμω`, `μήνε`, `εσά` instead of
`σας`, `όμως`, `μήνες`, `εσάς`.

**Cause.** A bug in the model itself, reported upstream in
[discussion #25](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/25) (unresolved).
We verified that:

- the tokenizer encodes and decodes `ς` correctly (token id 14855);
- it is not a special or suppressed token;
- the model ranks `ς` about 100th where it belongs, in both bf16 and fp32.

**Workaround (on by default for `-l el`).** `greek.py` appends `ς` to a word `w` when `w+ς` is at
least **20×** more frequent in Greek than `w`, using a 50k-word frequency list
(`resources/el_50k.txt`, FrequencyWords, CC BY-SA 4.0).

| Fixed | Left unchanged (ambiguous, both forms common) |
|---|---|
| σα → σας, όμω → όμως, μήνε → μήνες, εσά → εσάς | τη/της, του/τους, αυτό/αυτός, ένα/ένας |

So some ς will still be missing, mainly in articles and pronouns. Proofread important transcripts.
Disable the fix with `--no-sigma-fix`, or tune `DEFAULT_RATIO` in `greek.py`.

**If the model is updated upstream:** re-run `uv run speech-to-text download` and test with
`--no-sigma-fix`. If ς appears natively, the workaround can be removed.

## Repetition loops

**Symptom.** Occasionally the decoder gets stuck repeating a sentence, e.g. `Να κάτσουμε.` ×22
(seen once in a 78-minute recording).

**Workaround (always on).** `postprocess.collapse_repetitions` reduces any sentence repeated 4+
times in a row to 2 copies. Natural repeats such as "Ναι. Ναι. Ναι." are left alone.

## Other model limitations

- No automatic language detection.
- No word timestamps or speaker diarization. Timestamps are per ~30 s segment.
- Capitalisation/punctuation may be inconsistent across segment boundaries.
- Proper names and military/technical terms are often misheard. There is no prompt or vocabulary-boosting option.
