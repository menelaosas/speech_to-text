# Accuracy: what was tried and what works

All numbers come from `scripts/evaluate.py` on the Greek
[FLEURS](https://huggingface.co/datasets/google/fleurs) test set (read speech, 650 clips with
reference transcripts). Metrics: **WER** (word error rate) and **CER** (character error rate), computed
lowercase without punctuation and with accents kept.

## Summary

| Change | Effect | Status |
| --- | --- | --- |
| Beam search (`--beams 4`) | WER 9.15% → 8.86% on clean speech, removes some hallucinations | **default** |
| Context-aware ς restoration with XLM-R (`--sigma-fix lm`) | ς-related word errors −61% when the model drops ς | **default** |
| Adaptive ς margin | Avoids adding wrong ς when the model already writes them | **default** |
| ς logit bias during decoding | WER 9.2% → 9.7–13.9%: inserts ς in wrong places | rejected |
| Prompt context text (`<\|startofcontext\|>`) | Model copies the context into every segment and hallucinates | rejected |
| `<\|itn\|>`, `<\|emo:neutral\|>` prompt tokens | No ς improvement, changes content | rejected |
| Audio filters (loudnorm, dynaudnorm, denoise, band-pass) | No effect on ς, unclear effect on WER | not used |
| Segment length 8 / 15 / 30 s | No effect on ς | 30 s kept |
| fp32 instead of bf16 | Identical output, needs 8 GB VRAM | bf16 kept |

## The ς problem in detail

On **clean read speech** (FLEURS) the model writes ς normally: 17.8% of output words end in ς,
the same as the references. On **conversational recordings** (e.g. the meeting recording in `data/input/`) it writes
almost none (0.0%). The cause is in the model and is reported upstream
([discussion #25](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/25)).
Nothing on the input side (filters, segment length, prompt) changes it, so it is fixed after transcription.

### Restoration benchmark

Take the 350 held-out reference sentences (rows 300–649, not used for tuning), delete every final ς
(1,423 of 7,969 words), and measure how well each method restores them:

| Method | Restored | Wrongly added | Still missing | Word accuracy |
| --- | --- | --- | --- | --- |
| none | 0 | 0 | 1423 | 82.1% |
| `dict`: frequency ratio ≥ 20 | 540 | 9 | 883 | 88.8% |
| `lm`: XLM-R-large, margin −1 | 1181 | 106 | 242 | **95.6%** |

Also tried during tuning (rows 0–299): Qwen2.5-0.5B (90.1%), Qwen2.5-1.5B (91.4%), XLM-R-base (93.5%),
XLM-R-large with the 50k word list (93.8%) and with the full word list (95.3–96.0% depending on margin).

### How `lm` works

1. Candidates: every Greek word `w` where `w + ς` appears at least 3 times in the full FrequencyWords
   list (`resources/el_sigma_words.txt`, 101k words).
2. For each candidate, take ±8 words of context and compute XLM-RoBERTa-large's pseudo-log-likelihood
   of the word written both ways (each sub-token masked in turn).
3. Add ς if `score(w+ς) > score(w) + margin`.

### Adaptive margin

The best margin depends on whether the model dropped the ς:

| Raw transcript ς rate | Situation | Margin | Result |
| --- | --- | --- | --- |
| < 5% | model dropped ς (conversational audio) | −1 (lean towards ς) | 95.6% word acc. on ς-stripped text |
| ≥ 5% | model behaving (clean speech) | +3 (only confident fixes) | FLEURS WER 8.88% → 8.78% |

## Current defaults on FLEURS (200 clips, 36.7 min)

| Config | WER | CER | Time |
| --- | --- | --- | --- |
| greedy, no fix | 9.15% | 3.43% | 17 s |
| beams 4, dict | 8.86% | 3.28% | 29 s |
| **beams 4, lm (default)** | **8.78%** | **3.27%** | 36 s |

On the 78.5-minute meeting recording: 79 s in total (~60× real time). The share of words ending in ς
went from 5.0% (dict) to 10.5% (lm).

## Measured on the real recording

The operator transcribed 3.5 minutes of the meeting recording by hand (00:31:38–00:35:08, 571 words,
7 speakers). Scored with the pilot pipeline ([Pilot](Pilot.md)) and a documented Greek normaliser
(lowercase, no tonos/dialytika, ς→σ, numbers to words, no punctuation):

| Measured | Result |
| --- | --- |
| WER raw Cohere output | 42.0% |
| WER with ς restoration (`lm`) | 37.0% (30 fewer word errors) |
| BF16 (GPU) vs FP32 (CPU), same audio pieces | 44.1% vs 44.3%: no loss from BF16 |
| Beam 4 vs greedy | 39.1% vs 39.6% |
| Long pieces (VAD, ≤30 s) vs short speaker turns | 35.9% vs 38.9%: short pieces lose words |

So on conversational audio, about 4 in 10 words differ from the hand transcript. About 22 of the 561
reference words are typing slips, so part of that comes from the reference. The ς restoration is the
largest single gain measured on this audio.

## Caveats

- FLEURS is clean read speech; the meeting recording scores far worse (37% vs 8.8% WER). The real
  recording has been measured on **one** 3.5-minute excerpt only.
- FLEURS sentences come from Wikipedia, which XLM-R may have seen in training, so the ς numbers may
  be slightly optimistic.
- Names that aren't in the word list never get a ς (e.g. "Βαδριανό").

## Ideas not yet implemented

- **A second reference excerpt** of the real recording, to confirm the numbers above (done once; see
  [Pilot](Pilot.md)).
- **Fine-tuning** (LoRA) of Cohere Transcribe on Greek conversational speech with correct ς. This
  would fix the root cause but needs labelled data and a lot of work.
- **LLM proof-reading** of the transcript for names and terms. Strong, but sending transcripts to a
  cloud API may not be acceptable for sensitive recordings. A local LLM would need a bigger GPU.
- ~~Voice activity detection~~, ~~speaker diarization~~, ~~word timestamps~~: implemented in the
  [pilot](Pilot.md).
