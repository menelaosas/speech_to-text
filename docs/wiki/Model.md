# The model: Cohere Transcribe (03-2026)

- **Hub id**: [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- **Size**: 2B parameters, bf16 weights (`model.safetensors`, ~4.1 GB)
- **License**: Apache 2.0, gated (accept terms on Hugging Face)
- **Architecture**: encoder-decoder ASR, 128-bin log-mel features at 16 kHz, 16k SentencePiece vocabulary
- **Transformers support**: native `CohereAsrForConditionalGeneration` since transformers 5.4.
  No `trust_remote_code` is needed. The repo's own remote code does **not** load with transformers 5.20.

## Languages (14)

English `en`, French `fr`, German `de`, Italian `it`, Spanish `es`, Portuguese `pt`, Greek `el`,
Dutch `nl`, Polish `pl`, Mandarin `zh`, Japanese `ja`, Korean `ko`, Vietnamese `vi`, Arabic `ar`.

There is **no language detection**. You must pass the language.

## Decoder prompt

The processor builds the prompt
`<|startofcontext|><|startoftranscript|><|emo:undefined|><|el|><|el|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`.
`punctuation=False` switches `<|pnc|>` to `<|nopnc|>`. The model does not produce timestamps or speaker labels.
Our timestamps come from the segment boundaries.

## Limits

- Max clip: 35 s (we use 30 s segments)
- `max_new_tokens=256` per segment, which is plenty for 30 s of speech

## Performance (measured on this project)

RTX 4070 Laptop 8 GB, bf16, batch 8, 78.5 min recording (172 segments):

| Settings | Time | Speed |
|---|---|---|
| greedy, dictionary ς fix | 29 s | ~160× real time |
| beams 4 + XLM-R ς fix (default) | 79 s | ~60× real time |

Accuracy on Greek FLEURS: WER 8.8%, CER 3.3% (see [Accuracy](Accuracy.md)).
