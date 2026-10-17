"""Browser API integration: uploads, seekable audio, immutable originals and durable edits."""

from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from speech_pilot.config import DEFAULT_CONFIG
from speech_pilot.web import Workspace, make_server, validate_edits


def transcript():
    return {'source': {'path': '/private/ηχογράφηση.wav', 'duration_s': 5}, 'language': 'el',
            'segments': [{'id': 's1', 'start': 1.25, 'end': 3.5, 'text': 'Καλημέρα κόσμε',
                          'primary_speaker': 'SPEAKER_00', 'flags': [], 'words': [
                              {'text': 'Καλημέρα', 'start': 1.25, 'end': 2.5, 'aligned': True},
                              {'text': 'κόσμε', 'start': 2.5, 'end': 3.5, 'aligned': True}]}]}


def edits():
    return {'segments': [{'id': 's1', 'speaker': 'Μαρία', 'words': ['Καλησπέρα', 'σε όλους'], 'text': 'ignored'}]}


@contextmanager
def running(tmp_path):
    ws = Workspace(tmp_path / 'workspace', DEFAULT_CONFIG)
    server = make_server(ws, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None):
        connection = HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
        connection.request(method, path, body, {'X-Review-Client': '1', **(headers or {})})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    try:
        yield ws, request
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        ws.pool.shutdown()


def ready_job(ws):
    job_id, _ = ws.create('ηχογράφηση.wav')
    folder = ws.folder(job_id)
    (folder / 'run' / '02_prepare').mkdir(parents=True)
    (folder / 'run' / '02_prepare' / 'processing.wav').write_bytes(b'RIFF0123456789WAVE')
    (folder / 'run' / 'transcript.json').write_text(json.dumps(transcript(), ensure_ascii=False))
    ws.update(job_id, status='ready')
    return job_id


def test_review_save_reopen_export_and_audio_range(tmp_path):
    with running(tmp_path) as (ws, request):
        job_id = ready_job(ws)
        original = (ws.folder(job_id) / 'run' / 'transcript.json').read_bytes()
        base = f'/api/jobs/{job_id}'
        assert request('GET', '/')[0] == 200
        for asset in ('/app.js', '/style.css'):
            assert request('GET', asset)[0] == 200
        assert json.loads(request('GET', base + '/transcript')[2])['draft'] is None
        status, headers, body = request('GET', base + '/audio', headers={'Range': 'bytes=4-8'})
        assert (status, body, headers['Content-Range']) == (206, b'01234', 'bytes 4-8/18')
        assert request('GET', base + '/audio', headers={'Range': 'bytes=999-'})[0] == 416
        status, _, body = request('POST', base + '/save', json.dumps(edits()).encode())
        assert status == 200
        version = json.loads(body)['version']
        reloaded = json.loads(request('GET', base + '/transcript')[2])
        assert reloaded['draft']['segments'][0]['text'] == 'Καλησπέρα σε όλους'
        assert reloaded['transcript'] == transcript()
        for kind in ('txt', 'srt', 'json'):
            status, _, body = request('GET', base + f'/download?version={version}&format={kind}')
            assert status == 200 and 'Μαρία' in body.decode() and 'Καλησπέρα' in body.decode()
            if kind == 'srt':
                assert '00:00:01,250 --> 00:00:03,500' in body.decode()
            if kind == 'json':
                assert json.loads(body)['original'] == transcript()
        assert (ws.folder(job_id) / 'run' / 'transcript.json').read_bytes() == original
        other = json.loads(request('POST', base + '/save', json.dumps(edits()).encode())[2])['version']
        assert other != version


def test_upload_queues_and_preserves_audio(tmp_path):
    with running(tmp_path) as (ws, request):
        seen = threading.Event()
        with patch.object(ws, 'run', lambda job: seen.set()):
            status, _, body = request('POST', '/api/jobs?name=interview.m4a', b'audio-content')
            assert status == 202
            job = json.loads(body)
            assert seen.wait(2)
            assert (ws.folder(job['id']) / 'source.m4a').read_bytes() == b'audio-content'
            assert job['status'] == 'queued'
        assert request('POST', '/api/jobs?name=script.py', b'bad')[0] == 400
        assert request('POST', '/api/jobs?name=empty.wav', b'')[0] == 400


def test_local_origin_and_path_boundaries(tmp_path):
    with running(tmp_path) as (ws, request):
        assert request('GET', '/', headers={'Host': 'attacker.example'})[0] == 403
        assert request('POST', '/api/jobs?name=a.wav', b'x', {'Origin': 'https://attacker.example'})[0] == 403
        assert request('POST', '/api/jobs?name=a.wav', b'x', {'X-Review-Client': ''})[0] == 403
        assert request('GET', '/api/jobs/../transcript')[0] == 404
        job_id = ready_job(ws)
        assert request('GET', f'/api/jobs/{job_id}/download?version=../../source&format=txt')[0] == 400


def test_invalid_edits_cannot_change_alignment():
    bad = edits()
    bad['segments'][0]['words'].append('extra')
    with pytest.raises(ValueError, match='Word count'):
        validate_edits(transcript(), bad)
    bad = edits()
    bad['segments'][0]['id'] = 'other'
    with pytest.raises(ValueError, match='IDs'):
        validate_edits(transcript(), bad)
    bad = edits()
    bad['segments'][0]['speaker'] = 'a\nb'
    with pytest.raises(ValueError, match='speaker'):
        validate_edits(transcript(), bad)


def test_pipeline_uses_existing_cli_and_resolved_config(tmp_path):
    ws = Workspace(tmp_path, DEFAULT_CONFIG)
    job_id, source = ws.create('recording.wav')
    source.write_bytes(b'audio')
    def fake_run(command, **kwargs):
        assert command[1:3] == ['-m', 'speech_pilot.cli']
        config = json.loads(Path(command[4]).read_text())
        assert config['asr']['segmentation'] == 'aligned'
        assert config['input']['path'] == str(source)
        assert config['input']['excerpt'] is None
        assert config['reference']['reuse_dir'] is None
        assert Path(config['diarization']['model_path']).is_absolute()
        output = ws.folder(job_id) / 'run'
        output.mkdir()
        (output / 'transcript.json').write_text(json.dumps(transcript()))
        return type('Result', (), {'returncode': 0})()
    try:
        with patch('speech_pilot.web.subprocess.run', fake_run):
            ws.run(job_id)
        assert ws.read(job_id)['status'] == 'ready'
    finally:
        ws.pool.shutdown()


def test_import_and_restart_preserve_draft(tmp_path):
    source = tmp_path / 'existing'
    (source / '02_prepare').mkdir(parents=True)
    (source / '02_prepare' / 'processing.wav').write_bytes(b'RIFFtest')
    (source / '02_prepare' / 'prepare.json').write_text('{"offset_s": 60}')
    original = json.dumps(transcript())
    (source / 'transcript.json').write_text(original)
    with running(tmp_path) as (ws, request):
        job_id = ws.import_run(source)
        root = ws.root
        assert json.loads(request('GET', f'/api/jobs/{job_id}/transcript')[2])['offset'] == 60
        assert request('POST', f'/api/jobs/{job_id}/save', json.dumps(edits()).encode())[0] == 200
        pending, _ = ws.create('pending.wav')
    restarted = Workspace(root, DEFAULT_CONFIG)
    try:
        assert restarted.read(job_id)['status'] == 'ready'
        assert (restarted.folder(job_id) / 'draft.json').exists()
        assert restarted.read(pending)['status'] == 'failed'
        assert (source / 'transcript.json').read_text() == original
    finally:
        restarted.pool.shutdown()


def test_original_download_retry_delete_and_assets(tmp_path):
    with running(tmp_path) as (ws, request):
        job_id = ready_job(ws)
        base = f'/api/jobs/{job_id}'
        # machine transcript is downloadable without saving a version; raw text, primary speaker
        status, _, body = request('GET', base + '/download?version=original&format=txt')
        assert status == 200 and 'SPEAKER_00: Καλημέρα κόσμε' in body.decode()
        assert json.loads(request('GET', base + '/download?version=original&format=json')[2]) == transcript()
        assert request('GET', base + '/download?version=original&format=exe')[0] == 400
        # latest saved version is reported so the download menu survives a reload
        assert json.loads(request('GET', base + '/transcript')[2])['latest_version'] is None
        version = json.loads(request('POST', base + '/save', json.dumps(edits()).encode())[2])['version']
        assert json.loads(request('GET', base + '/transcript')[2])['latest_version'] == version
        # favicon served, legacy /favicon.ico answered without a 404
        status, headers, _ = request('GET', '/favicon.svg')
        assert status == 200 and headers['Content-Type'] == 'image/svg+xml'
        assert request('GET', '/favicon.ico')[0] == 204
        # retry only for failed jobs, delete only when not processing
        assert request('POST', base + '/retry')[0] == 400
        ws.update(job_id, status='running')
        assert request('POST', base + '/delete')[0] == 400
        ws.update(job_id, status='failed', error='boom')
        assert b'missing' in request('POST', base + '/retry')[2]  # no uploaded file: upload again
        (ws.folder(job_id) / 'source.wav').write_bytes(b'audio')
        with patch.object(ws, 'run', lambda job: None):
            status, _, body = request('POST', base + '/retry')
        assert status == 202 and json.loads(body)['status'] == 'queued'
        ws.update(job_id, status='ready')
        assert request('POST', base + '/delete', headers={'X-Review-Client': ''})[0] == 403
        assert request('POST', base + '/delete')[0] == 200
        assert not (ws.root / job_id).exists()
        assert request('GET', base)[0] == 404


def test_integrity_step_counts_as_completed(tmp_path):
    ws = Workspace(tmp_path, DEFAULT_CONFIG)
    try:
        job_id = ready_job(ws)
        run = ws.folder(job_id) / 'run'
        (run / '08_export').mkdir()
        (run / '08_export' / 'step.json').write_text('{}')
        (run / '09_integrity').mkdir()
        (run / '09_integrity' / 'integrity.json').write_text('{}')
        assert ws.read(job_id)['completed_steps'][-2:] == ['08_export', '09_integrity']
    finally:
        ws.pool.shutdown()
