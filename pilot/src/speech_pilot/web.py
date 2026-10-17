"""Loopback browser workspace; model execution stays in the existing CLI process."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from .config import DEFAULT_CONFIG, PILOT_ROOT, load_config
from .export import hms, srt_time

ASSETS = Path(__file__).parent / 'web_assets'
MAX_UPLOAD = 2 * 1024**3
AUDIO_TYPES = {'.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.flac', '.aac', '.opus', '.webm', '.aiff', '.wma'}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_edits(transcript, edits):
    """Accept text/speaker corrections only; timestamps and provenance remain server-owned."""
    if not isinstance(edits, dict) or set(edits) != {'segments'}:
        raise ValueError('Expected a segments correction list')
    rows = edits['segments']
    if not isinstance(rows, list) or len(rows) != len(transcript['segments']):
        raise ValueError('All original segments must be present')
    clean = []
    for original, row in zip(transcript['segments'], rows):
        if not isinstance(row, dict) or row.get('id') != original['id']:
            raise ValueError('Segment IDs or order changed')
        speaker = row.get('speaker')
        words = row.get('words')
        if not isinstance(speaker, str) or not 1 <= len(speaker.strip()) <= 100 or '\n' in speaker or '\r' in speaker:
            raise ValueError('Enter a speaker label between 1 and 100 characters')
        expected = original.get('words') or []
        if not isinstance(words, list) or len(words) != len(expected):
            raise ValueError('Word count changed; correct text within its original time slot')
        if any(not isinstance(w, str) or len(w) > 2000 or '\n' in w or '\r' in w for w in words):
            raise ValueError('Invalid word correction')
        text = row.get('text')
        if not isinstance(text, str) or len(text) > 100000:
            raise ValueError('Invalid segment text')
        clean.append({'id': original['id'], 'speaker': speaker.strip(), 'words': words,
                      'text': ' '.join(w for w in words if w.strip()) if expected else text})
    return {'segments': clean}


def render_export(transcript, edits, kind):
    if kind == 'json':
        return json.dumps({'format': 'speech-review-1', 'original': transcript, 'corrections': edits},
                          ensure_ascii=False, indent=2)
    lines = []
    for i, (segment, edit) in enumerate(zip(transcript['segments'], edits['segments']), 1):
        if kind == 'srt':
            lines.append(f"{i}\n{srt_time(segment['start'])} --> {srt_time(segment['end'])}\n[{edit['speaker']}] {edit['text']}\n")
        else:
            lines.append(f"[{hms(segment['start'])} – {hms(segment['end'])}] {edit['speaker']}: {edit['text']}")
    return '\n'.join(lines) + '\n'


class Workspace:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = Path(config).resolve()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        for path in self.root.glob('*/job.json'):
            job = json.loads(path.read_text())
            if job['status'] in ('queued', 'running', 'uploading'):
                job.update(status='failed', error='Server stopped before this job completed. Upload again to retry.')
                atomic_json(path, job)

    def folder(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise FileNotFoundError('Unknown recording')
        folder = self.root / job_id
        if not (folder / 'job.json').exists():
            raise FileNotFoundError('Unknown recording')
        return folder

    def read(self, job_id):
        with self.lock:
            folder = self.folder(job_id)
            job = json.loads((folder / 'job.json').read_text())
        run = folder / 'run'
        job['completed_steps'] = [p.parent.name for p in sorted(run.glob('0*/step.json'))]
        if (folder / 'pipeline.log').exists():
            with (folder / 'pipeline.log').open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 8000))
                job['log'] = log.read().decode('utf-8', errors='replace')
        return job

    def update(self, job_id, **values):
        with self.lock:
            path = self.folder(job_id) / 'job.json'
            job = json.loads(path.read_text())
            job.update(values)
            atomic_json(path, job)

    def create(self, name):
        name = Path(name.replace('\\', '/')).name
        extension = Path(name).suffix.lower()
        if extension not in AUDIO_TYPES:
            raise ValueError('Choose a supported audio file (WAV, MP3, M4A, FLAC, OGG, or WebM)')
        job_id = uuid4().hex
        folder = self.root / job_id
        folder.mkdir()
        atomic_json(folder / 'job.json', {'id': job_id, 'name': name, 'status': 'uploading',
                                         'created_at': datetime.now(timezone.utc).isoformat(),
                                         'input': 'source' + extension})
        return job_id, folder / ('source' + extension)

    def run(self, job_id):
        folder = self.folder(job_id)
        try:
            self.update(job_id, status='running')
            job = self.read(job_id)
            cfg = load_config(self.config, {'input.path': str(folder / job['input']),
                'input.excerpt': None, 'reference.reuse_dir': None, 'asr.segmentation': 'aligned',
                'run.output_root': str(folder)})
            config_path = folder / 'config.json'
            atomic_json(config_path, cfg.model_dump(mode='json'))  # JSON is also valid YAML.
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            with (folder / 'pipeline.log').open('wb') as log:
                result = subprocess.run([sys.executable, '-m', 'speech_pilot.cli', '-c', str(config_path),
                                         'run', '--run-id', 'run'], stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode or not (folder / 'run' / 'transcript.json').exists():
                raise RuntimeError('Transcription failed. See the processing log for model or device details.')
            self.update(job_id, status='ready')
        except Exception as exc:
            self.update(job_id, status='failed', error=str(exc))

    def import_run(self, source):
        source = Path(source).resolve()
        transcript = json.loads((source / 'transcript.json').read_text())
        wav = source / '02_prepare' / 'processing.wav'
        if not wav.exists():
            raise ValueError('Existing run must include 02_prepare/processing.wav')
        job_id, audio = self.create('Imported recording.wav')
        folder = self.folder(job_id)
        run = folder / 'run'
        (run / '02_prepare').mkdir(parents=True)
        shutil.copyfile(wav, run / '02_prepare' / 'processing.wav')
        shutil.copyfile(source / 'transcript.json', run / 'transcript.json')
        prep = source / '02_prepare' / 'prepare.json'
        if prep.exists():
            shutil.copyfile(prep, run / '02_prepare' / 'prepare.json')
        self.update(job_id, name=Path(transcript['source']['path']).name, status='ready')
        return job_id


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body, content_type='application/json', status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), status=status)

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        try:
            # Reject DNS rebinding and cross-origin writes to this local-only service.
            if self.headers.get('Host') not in self.server.allowed_hosts:
                return self.json({'error': 'Invalid host'}, 403)
            if self.command == 'POST' and (self.headers.get('Origin') not in (None, *self.server.origins)
                    or self.headers.get('X-Review-Client') != '1'):
                return self.json({'error': 'Open the app from its local URL'}, 403)
            self.route()
        except FileNotFoundError:
            self.json({'error': 'Recording or file not found'}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            self.close_connection = True
            self.json({'error': str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            self.json({'error': 'Server error; see the terminal for details'}, 500)

    def body_length(self, maximum):
        length = int(self.headers.get('Content-Length', '-1'))
        if not 0 < length <= maximum or self.headers.get('Transfer-Encoding'):
            raise ValueError(f'Request size must be between 1 and {maximum} bytes')
        return length

    def route(self):
        ws = self.server.workspace
        url = urlsplit(self.path)
        parts = url.path.strip('/').split('/')
        if self.command == 'GET' and url.path in ('/', '/app.js', '/style.css'):
            name = 'index.html' if url.path == '/' else parts[0]
            return self.send_bytes((ASSETS / name).read_bytes(), mimetypes.guess_type(name)[0] + '; charset=utf-8')
        if parts == ['api', 'jobs']:
            if self.command == 'GET':
                jobs = [ws.read(p.parent.name) for p in ws.root.glob('*/job.json')]
                return self.json(sorted(jobs, key=lambda j: j['created_at'], reverse=True))
            length = self.body_length(MAX_UPLOAD)
            job_id, path = ws.create(parse_qs(url.query).get('name', ['audio.wav'])[0])
            try:
                with path.open('wb') as target:
                    while length:
                        chunk = self.rfile.read(min(length, 1024 * 1024))
                        if not chunk:
                            raise ValueError('Upload interrupted')
                        target.write(chunk)
                        length -= len(chunk)
                ws.update(job_id, status='queued')
                ws.pool.submit(ws.run, job_id)
            except Exception:
                ws.update(job_id, status='failed', error='Upload interrupted')
                raise
            return self.json(ws.read(job_id), 202)
        if len(parts) < 3 or parts[:2] != ['api', 'jobs']:
            raise FileNotFoundError()
        job_id = parts[2]
        folder = ws.folder(job_id)
        if len(parts) == 3 and self.command == 'GET':
            return self.json(ws.read(job_id))
        action = parts[3] if len(parts) == 4 else ''
        if action == 'audio' and self.command == 'GET':
            return self.audio(folder / 'run' / '02_prepare' / 'processing.wav')
        transcript = json.loads((folder / 'run' / 'transcript.json').read_text())
        if action == 'transcript' and self.command == 'GET':
            prep = folder / 'run' / '02_prepare' / 'prepare.json'
            offset = json.loads(prep.read_text()).get('offset_s', 0) if prep.exists() else 0
            draft = folder / 'draft.json'
            return self.json({'transcript': transcript, 'offset': offset,
                              'draft': json.loads(draft.read_text()) if draft.exists() else None})
        if action == 'save' and self.command == 'POST':
            data = json.loads(self.rfile.read(self.body_length(10 * 1024**2)))
            edits = validate_edits(transcript, data)
            version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
            with ws.lock:
                exports = folder / 'edits'
                exports.mkdir(exist_ok=True)
                for kind in ('json', 'txt', 'srt'):
                    (exports / f'{version}.{kind}').write_text(render_export(transcript, edits, kind), encoding='utf-8')
                atomic_json(folder / 'draft.json', edits)
            return self.json({'version': version})
        if action == 'download' and self.command == 'GET':
            query = parse_qs(url.query)
            version, kind = query.get('version', [''])[0], query.get('format', ['txt'])[0]
            if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}', version) or kind not in ('json', 'txt', 'srt'):
                raise ValueError('Invalid export')
            return self.send_bytes((folder / 'edits' / f'{version}.{kind}').read_bytes(),
                                   'application/json' if kind == 'json' else 'text/plain; charset=utf-8')
        raise FileNotFoundError()

    def audio(self, path):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.send_bytes(b'', status=416)
            a, b = match.groups()
            if a:
                start, end = int(a), min(int(b), end) if b else end
            else:
                start = max(0, size - int(b))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as audio:
            audio.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = audio.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(workspace, port=8765):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.workspace = workspace
    port = server.server_address[1]
    server.allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    server.origins = {f'http://{host}' for host in server.allowed_hosts}
    return server


def main():
    parser = argparse.ArgumentParser(description='Local audio transcription and correction workspace')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-dir', type=Path, default=PILOT_ROOT / '.web-data')
    parser.add_argument('--import-run', type=Path, help='Copy a completed pilot run into the review workspace')
    args = parser.parse_args()
    ws = Workspace(args.data_dir, args.config)
    if args.import_run:
        ws.import_run(args.import_run)
    server = make_server(ws, args.port)
    print(f'Speech Studio: http://127.0.0.1:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nClosing server. Waiting for any active transcription to finish…', flush=True)
    finally:
        server.server_close()
        ws.pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
