# Usage

## Transcribe

```bash
uv run speech-to-text transcribe [INPUTS ...] [options]
```

`INPUTS` can be files or folders. With no inputs, every audio file in `data/input/` is processed.
Supported extensions: m4a, mp3, wav, flac, ogg, opus, aac, webm, mp4, mkv (anything ffmpeg reads).

| Option | Default | Meaning |
|---|---|---|
| `-l, --language` | `el` | Spoken language. **Required by the model**: it has no auto-detection. |
| `-o, --output-dir` | `data/output` | Where transcripts are written |
| `-f, --formats` | `txt srt` | Any of `txt`, `srt`, `json` |
| `--timestamps` | off | Prefix each paragraph in `.txt` with `[HH:MM:SS]` |
| `--no-punctuation` | off | Lowercase output without punctuation |
| `--no-sigma-fix` | off | Disable Greek final-sigma restoration ([Known Issues](Known-Issues.md)) |
| `-b, --batch-size` | `8` | Segments decoded per GPU batch. Lower it if you run out of memory. |
| `--model` | local copy | Model path or Hub id |
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `-v, --verbose` | off | Debug logging |

Language codes: `en fr de it es pt el nl pl zh ja ko vi ar`.

### Examples

```bash
uv run speech-to-text transcribe "data/input/ΓΕΝΙΚΟ ΕΠΙΤΕΛΕΙΟ ΕΘΝΙΚΗΣ ΑΜΥΝΑΣ 11.m4a"
uv run speech-to-text transcribe ~/podcasts/ -l en -f txt json -o ~/transcripts
uv run speech-to-text transcribe --device cpu -b 2      # no GPU
```

## Output formats

The output file is named after the input: `recording.m4a` → `recording.txt`, `recording.srt`, `recording.json`.

- **txt**: one paragraph per ~30 s segment. Use `--timestamps` for `[00:01:30] ...` lines.
- **srt**: subtitles with start/end times. Opens in VLC or any subtitle editor.
- **json**:
  ```json
  {"source": "recording.m4a", "language": "el", "text": "...",
   "segments": [{"start": 0.0, "end": 29.6, "text": "..."}]}
  ```

## Download

```bash
uv run speech-to-text download [--model-dir PATH]
```

## Using it from Python

```python
from pathlib import Path
from speech_to_text.audio import load_audio, split_audio
from speech_to_text.transcriber import Transcriber
from speech_to_text.writers import to_txt

t = Transcriber()
segments = split_audio(load_audio(Path("data/input/recording.m4a")))
result = t.transcribe_segments(segments, language="el")
print(to_txt(result))
```
