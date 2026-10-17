# Speech Studio (web UI)

A local browser app for the [pilot](Pilot.md) pipeline. You upload a recording, it is transcribed with
speakers, and then you listen with word-by-word highlighting while correcting words and speaker labels.
Corrections are saved as new versions; the machine transcript is never changed.

- Command: `speech-pilot-web` (in `pilot/`)
- Code: `pilot/src/speech_pilot/web.py` (server), `pilot/src/speech_pilot/web_assets/` (`index.html`,
  `app.js`, `style.css`)
- Tests: `pilot/tests/test_web.py`
- Full setup and controls: [pilot/README.md → Browser workspace](../../pilot/README.md#browser-workspace)

## Start

```bash
cd pilot
uv sync
uv run speech-pilot-web          # → Speech Studio: http://127.0.0.1:8765
```

Open **<http://127.0.0.1:8765>** and keep the terminal open while you work. No Node.js or build step is
needed. Stop with Ctrl+C; a transcription that is running is allowed to finish first.

| Option | Default | Use |
| --- | --- | --- |
| `--port` | `8765` | another port, e.g. `--port 9000` |
| `--config` | `config/pilot.yaml` | another pilot config |
| `--data-dir` | `pilot/.web-data` | where recordings, logs and corrections are stored |
| `--import-run` | – | copy a finished pilot run into the workspace, e.g. `--import-run runs/pilot-final` |

**Requirements:** the same as the pilot: all model folders from `config/pilot.yaml` downloaded
([Installation](Installation.md) step 5) and a CUDA GPU. Models run offline.

## Workflow

1. **Upload:** choose or drop an audio file (WAV, MP3, M4A, FLAC, OGG, WebM…, up to 2 GiB) and click
   **Transcribe recording**. A progress bar shows the upload.
2. **Processing:** a progress bar and the 11 pipeline steps (done ✓, current ◌, pending ○), elapsed time
   and the log. Jobs run one at a time. A 78-minute recording takes about 5–6 minutes on the RTX 4070.
   If it fails, the failed step is marked ✕ and the log opens; **Try again** re-runs it with the same
   upload (finished steps are reused) and **Delete recording** removes it.
3. **Listen:** the player stays pinned at the top while you scroll and shows who is speaking now. The
   current word is highlighted; each speaker has a colour; overlap segments list the other speakers
   ("with SPEAKER_02"). Click a word to jump there, or a segment's ▶ time to play from it.
   **Follow playback** can be switched off to scroll freely.
4. **Correct a word:** double-click it (or focus it and press **Enter**). The dialog marks the word in its
   context, shows what the machine wrote with **Restore**, and **Listen again** replays only that word.
   **Enter** applies. A correction can be a phrase, or blank to remove the word.
5. **Speakers:** the **Speakers** panel renames a speaker in every segment at once (SPEAKER_01 →
   "Πρόεδρος"). A single turn can still be changed in its own speaker field, which suggests existing names.
6. **Find what to check:** the search box highlights and counts matches. **Show** filters to *Needs
   review* (any flag, overlap included; 414 of 671 segments on the test recording) or *Warnings only*
   (flags other than overlap alone; 89). **Next to review** jumps to the next one below.
7. **Save and download:** **Save new version** is active when there are unsaved corrections. **Download**
   offers the *machine transcript* (as produced, without corrections) at any time, and *your saved
   version* once saved, each as Text, Subtitles (SRT) or JSON. The JSON keeps original and corrections
   apart. Saved corrections and the latest version reopen with the recording.

| Key | Action |
| --- | --- |
| `Space` | play / pause (outside text fields) |
| `←` / `→` | back / forward 5 s; on a focused word: previous / next word |
| `Enter` | correct the focused word; in the dialog: apply |
| `/` | jump to the search box |
| `Ctrl`/`⌘` + `S` | save a new version |
| `Esc` | close the dialog without changes |

A recording can be deleted from the editor (**Delete this recording…** at the bottom) or from the failed
view. This removes its workspace folder only, never files outside `.web-data`; processing jobs cannot be
deleted.

Words with a dotted underline have estimated timing (the aligner could not place them, e.g. Latin
letters), so highlighting there is approximate.

## How it fits the pipeline

```text
browser ──upload──► web.py (127.0.0.1 only) ──► python -m speech_pilot.cli run   (separate process)
   ▲                    │                           └─ same steps as the pilot: VAD, pyannote, Cohere,
   │                    │                              ς layer, word alignment, flags, export
   └──audio + words ◄───┴── .web-data/<job-id>/run/transcript.json  +  02_prepare/processing.wav
```

- The server never loads a model itself. It runs the normal pilot CLI as a subprocess, so the GPU rules,
  isolation and determinism of the pilot apply unchanged.
- The app's config is `config/pilot.yaml` with three settings forced: `asr.segmentation: aligned` (word
  timestamps are needed for highlighting), the whole file is processed, and `reference.reuse_dir: null`
  (the operator reference belongs to the original recording only and is never applied to uploads).
- Uploaded recordings are not scored: accuracy is only measured where a human reference exists.

## Where things are stored

Each recording gets `pilot/.web-data/<job-id>/` (ignored by git):

| Path | Content |
| --- | --- |
| `job.json` | name, status (`uploading` → `queued` → `running` → `ready` / `failed`), error |
| `source.<ext>` | the uploaded file, as received |
| `config.json` | the exact config used for this job |
| `pipeline.log` | full pipeline output (shown in the UI) |
| `run/` | a normal pilot run folder: `transcript.json`, step folders, report |
| `draft.json` | the last saved corrections (reopened in the editor) |
| `edits/<version>.txt/.srt/.json` | one set of files per **Save new version** |

`run/transcript.json` is never overwritten; every save creates new files. Uploads and WAV copies use
disk space: a 78-minute recording takes about 150 MB for the processing copy plus the upload itself.

## HTTP API (used by the page)

All routes are served by `web.py` on `127.0.0.1`. Every `POST` must carry `X-Review-Client: 1` and come
from the app's own origin.

| Method | Path | Does |
| --- | --- | --- |
| GET | `/`, `/app.js`, `/style.css`, `/favicon.svg` | the page and its assets |
| GET | `/api/jobs` | all recordings with status, finished steps and the log tail |
| POST | `/api/jobs?name=<file name>` | upload (body = audio bytes, ≤ 2 GiB); queues transcription → `202` |
| GET | `/api/jobs/<id>` | one recording's status, finished steps and log |
| GET | `/api/jobs/<id>/transcript` | transcript, audio offset, saved corrections (`draft`), `latest_version` |
| GET | `/api/jobs/<id>/audio` | the 16 kHz processing WAV, with `Range` support for seeking |
| POST | `/api/jobs/<id>/save` | validate and store corrections → new `version` (txt/srt/json files) |
| GET | `/api/jobs/<id>/download?version=<v>&format=txt\|srt\|json` | a saved version; `version=original` gives the machine transcript |
| POST | `/api/jobs/<id>/retry` | re-queue a **failed** job with its uploaded file (finished steps reused) |
| POST | `/api/jobs/<id>/delete` | delete a **ready or failed** recording's workspace folder |

Errors come back as `{"error": "..."}` with 400 (invalid request), 403 (wrong host/origin), 404
(unknown recording) or 500 (see the server terminal).

## What a correction can and cannot change

The server validates every save (`validate_edits`):

- **Can change:** a word's text (including into a phrase, or empty), and a turn's speaker label
  (1–100 characters).
- **Cannot change:** segment order or IDs, the number of words in a segment, or any timestamp. A corrected
  word keeps its original time slot and is not re-aligned.

The JSON export keeps the machine output and the human corrections separate
(`{"format": "speech-review-1", "original": …, "corrections": …}`), which preserves the record of what
the machine produced.

## Security

It is a single-user tool for this computer:

- It listens on `127.0.0.1` only, so other machines on the network cannot reach it.
- Requests with a different `Host` header are rejected (DNS-rebinding protection). Writes must come from the app's
  own origin and carry the `X-Review-Client` header, otherwise they get 403.
- Content-Security-Policy `default-src 'self'` and `nosniff` are sent; there are no external scripts or fonts.
- Upload size is limited (2 GiB); job IDs and export versions are strictly validated, so paths outside
  the workspace cannot be requested.
- There is **no login**: anyone using this computer account can open the app. Don't expose it with port
  forwarding or a reverse proxy.

## Review an existing run

```bash
uv run speech-pilot-web --import-run runs/pilot-final
```

This copies the transcript and the processing WAV into the workspace without running models; the
original run is not changed. Import once. Runs made before word timestamps (e.g. `runs/final`) only
support segment highlighting and whole-segment editing.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Address already in use` | another instance is running, or use `--port 9000` |
| Job ends as **failed** | the failed step is marked and the log opens. Usual causes: a model folder is missing, CUDA is unavailable, or the GPU ran out of memory (the pilot never falls back to CPU silently). Fix the cause, then **Try again** (no new upload needed) |
| Job shows **failed: Server stopped before this job completed** | the server was stopped during processing; click **Try again** |
| `Open the app from its local URL` (403) | open it at `http://127.0.0.1:<port>` or `http://localhost:<port>`, not through another host name |
| Unsupported file type | convert to WAV/MP3/M4A/FLAC/OGG/WebM first |
| No word highlighting | the recording was imported from a run without word timestamps; upload it again |

## Tests

```bash
cd pilot && uv run pytest tests/test_web.py -q
```

They cover saving, reopening and exporting corrections, audio range requests (seeking), upload queueing,
host/origin checks and path limits, rejected edits that would change alignment, the subprocess pipeline
call with the resolved config, import plus restart behaviour, machine-transcript download, the latest
version after reload, retry (only for failed jobs, needs the upload), delete (not while processing),
the favicon, and the integrity step counting as finished.

The interface itself was checked in headless Chrome (desktop 1440×900 and phone 390×844) on the
78-minute recording: no console errors, 60 fps while playing with ~11,000 words, editor opens in
~0.4 s, no horizontal scrolling on the phone.
