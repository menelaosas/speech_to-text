# Benchmarks: measuring improvements

Use the benchmarks to check whether a change to the pipeline really improves WER/CER. They run the
pipeline's word-producing steps on public Greek datasets with known transcripts:

- the same audio preparation as step 02 (ffmpeg → mono, 16 kHz, 16-bit);
- Cohere ASR with the configured decoding (beams, dtype…), long clips split exactly like the pipeline;
- the Greek ς layer.

Each run's result comes with a 95% confidence interval, and two runs can be compared clip by clip.

**What it does not measure:** speaker attribution and diarization (every clip has one speaker), or
accuracy on meeting recordings. For those, use the operator's reference excerpt ([Pilot](Pilot.md)).

## Datasets

| Dataset | Location (git-ignored) | Split | Clips | Audio | Speakers |
| --- | --- | --- | --- | --- | --- |
| Common Voice 27.0 Greek (`cv`) | `data/eval/common_voice/cv-corpus-27.0-2026-09-11/el/` | dev | 1,841 | 2.0 h | 50 |
| | | test | 1,842 | 2.2 h | 323 |
| FLEURS Greek (`fleurs`) | `data/eval/fleurs/parquet-data/el_gr/test-00000-of-00001.parquet` | test | 650 | 1.9 h | – |

The Common Voice folder is the release as downloaded (`clips/*.mp3`, `train/dev/test.tsv`,
`validated.tsv`, `clip_durations.tsv`, README). Paths are set in `pilot/config/pilot.yaml` → `benchmark`.

**Rule: try changes on `dev`, confirm on `test`.** Tuning on the test split makes its numbers
optimistic. Keep `test` for the final check of a change.

## Run

```bash
cd pilot
uv run speech-pilot benchmark --dataset cv --split dev --name baseline-cv-dev
uv run speech-pilot benchmark --dataset cv --split dev --name my-change --set asr.num_beams=8
uv run speech-pilot benchmark-compare runs/benchmarks/baseline-cv-dev runs/benchmarks/my-change
```

| Option | Meaning |
| --- | --- |
| `--dataset cv\|fleurs` | which dataset |
| `--split dev\|test\|train` | Common Voice split (FLEURS is always test) |
| `--limit N` | seeded random subset of N clips; the same clips every time (quick checks) |
| `--name` | output folder under `pilot/runs/benchmarks/` |
| `--set key=value` | any pipeline config override, e.g. `asr.num_beams=1`, `normalization.greek_final_sigma=false` |

A full Common Voice split takes about 1 minute on the RTX 4070 (audio decoding ~9 s, ASR ~40 s, ς layer
a few seconds).

### Output (`pilot/runs/benchmarks/<name>/`)

| File | Content |
| --- | --- |
| `REPORT.md` | WER/CER with 95% intervals, substitutions/deletions/insertions, timing and VRAM, the 15 worst clips |
| `results.json` | the same plus the full config, config digest, model revisions, dataset fingerprint (TSV/parquet SHA-256, hash of the chosen clip IDs) |
| `utterances.jsonl` | per clip: reference, raw output, ς-layer output, error counts |
| `compare_vs_<A>.json` | written by `benchmark-compare` in run B |

Scoring uses the same Greek normaliser as the reference scoring: lowercase, no tonos/dialytika, ς→σ,
numbers written as words, punctuation removed.

## Reading a comparison

`benchmark-compare A B` pairs the two runs clip by clip and reports **B minus A**:

```text
raw    WER  A 9.18%  B 8.92%  Δ -0.26 pts  95% CI [-0.46, -0.08]  → B better  (better on 34, worse on 21 clips)
```

- **B better / B worse** only when the whole 95% interval is on one side of zero.
- **No significant difference** means the change can't be told apart from chance on this data. Don't
  adopt it on the strength of the number alone.
- Both runs must cover exactly the same clips (same dataset, split and `--limit`); otherwise the command
  refuses to compare.

## Baselines (2026-10-17, current default config)

Cohere Transcribe `d552dadc`, BF16, 4 beams, ς layer with adaptive margin.

| Run | WER raw | 95% CI | WER ς layer | CER raw | ASR time |
| --- | --- | --- | --- | --- | --- |
| `baseline-cv-dev` | **8.92%** | 8.24–9.66 | 9.14% | 2.45% | 36 s |
| `baseline-cv-test` | **9.77%** | 9.06–10.51 | 10.09% | 2.74% | 40 s |
| `baseline-fleurs-test` | **7.42%** | 6.84–8.00 | 7.41% | 2.51% | 52 s |
| `greedy-cv-dev` (`asr.num_beams=1`) | 9.18% | 8.46–9.95 | 9.40% | 2.54% | 26 s |

First findings:

- **Beam search is confirmed:** 4 beams beat greedy by 0.26 points on CV dev (95% CI −0.46 to −0.08).
  The 3.5-minute meeting excerpt was too small to show this.
- **On read speech the ς layer is neutral to slightly worse** (+0.2–0.3 points on Common Voice). Cohere
  already writes ς there, so the layer can only add a few wrong ones. On the meeting recording, where the
  model drops ς, the layer is the largest gain measured (WER 42.0% → 37.0%). A candidate improvement is to
  skip the layer when the raw text already has a normal ς rate. That can be tested with
  `--set normalization.greek_final_sigma=false` and compared.

## Caveats

- Common Voice and FLEURS are **read speech**: clearer than meetings, so their WER (7–10%) is far lower
  than on the meeting excerpt (37%). Use them to compare changes, not to predict meeting accuracy.
- Cohere reports results on Common Voice 17 but does not publish its training data. Some Common Voice
  sentences may have been seen in training, so absolute scores may be optimistic. Comparisons between
  pipeline variants are still valid.
- A change that helps read speech can still hurt meetings, and vice versa: also check the meeting
  reference excerpt (`scripts/compare_on_reference.py`).
