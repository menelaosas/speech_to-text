"""Compare ordinary beams and experimental consensus on a saved human-reference window.

No diarization/alignment rerun, model downloads, or changes to the source run. The reference is used
only for scoring, never passed to the recognizer. This is a development experiment on one recording,
not a held-out benchmark or evidence that a gain generalizes.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import soundfile as sf

from speech_pilot.config import DEFAULT_CONFIG, load_config
from speech_pilot.gpu import free_gpu
from speech_pilot.pipeline import set_determinism
from speech_pilot.providers.base import AsrUnit
from speech_pilot.providers.cohere_asr import CohereASR
from speech_pilot.providers.xlmr_sigma import XlmrSigmaRestorer
from speech_pilot.score import parse_reference, score_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parakeet-model', type=Path, help='Also compare a locally downloaded Parakeet TDT v3')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    cfg = load_config(args.config)
    run = args.run.resolve()
    reference = run / 'reference'
    excerpt = json.loads((reference / 'excerpt.json').read_text())
    prep = json.loads((run / '02_prepare' / 'prepare.json').read_text())
    original = json.loads((run / 'transcript.json').read_text())
    a, b = excerpt['start_s'], excerpt['end_s']
    offset = prep.get('offset_s', 0)
    with sf.SoundFile(run / '02_prepare' / 'processing.wav') as source:
        sr = source.samplerate
        if sr != 16000 or source.channels != 1:
            raise ValueError('Expected the pilot 16 kHz mono processing WAV')
        if a < offset or b - offset > len(source) / sr + .01:
            raise ValueError('Human reference is outside the prepared recording')
        start_frame = round((a - offset) * sr)
        source.seek(start_frame)
        audio = source.read(round((b - a) * sr), dtype='float32')
    rows = json.loads((run / '05_asr' / 'asr_raw.json').read_text())['units']
    units = []
    for row in sorted(rows, key=lambda r: r['start']):
        start, end = max(a, row['start'] + offset), min(b, row['end'] + offset)
        if end <= start:
            continue
        lo, hi = round((start - a) * sr), round((end - a) * sr)
        units.append(AsrUnit(row['id'], start, end, audio[lo:hi]))
    if not units:
        raise ValueError('No ASR units overlap the reference window')
    text = ' '.join(t for _, t in parse_reference(reference / 'excerpt.txt'))
    if not text.strip():
        raise ValueError('A completed human reference is required')
    out = args.output or cfg.run.output_root / f'decoding-{datetime.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True, exist_ok=False)
    results = {'source_run': str(run), 'excerpt': {'start_s': a, 'end_s': b},
               'note': 'Development comparison on a previously inspected reference, not held-out performance.',
               'units': len(units), 'variants': {}}
    outputs = {}
    variants = [('best', {'num_candidates': 1, 'candidate_selection': 'best'}),
                          ('consensus', {'num_candidates': min(4, cfg.asr.num_beams),
                                         'candidate_selection': 'consensus'})]
    if args.parakeet_model:
        variants.append(('parakeet', {}))
    for name, changes in variants:
        if name == 'consensus' and cfg.asr.num_beams < 2:
            raise ValueError('Use at least two beams for the consensus comparison')
        set_determinism(cfg.run.seed)
        if name == 'parakeet':
            from speech_pilot.providers.parakeet_asr import ParakeetASR
            provider = ParakeetASR(args.parakeet_model, cfg.device.name)
        else:
            provider = CohereASR(cfg.asr.model_copy(update=changes), cfg.device.name)
        started = time.perf_counter()
        try:
            provider.load()
            outputs[name] = provider.transcribe(units, original['language'])
            results['variants'][name] = {'decoding': provider.decoding(), 'model': provider.info().model_dump(),
                'elapsed_s': round(time.perf_counter() - started, 2), 'events': provider.events,
                'units': [dict(unit_id=o.unit_id, text=o.text, tokens=o.tokens, generation=o.generation)
                          for o in outputs[name]]}
        finally:
            provider.unload()
            free_gpu()
    normalizer = XlmrSigmaRestorer(cfg.normalization, cfg.device.name) if cfg.normalization.greek_final_sigma else None
    try:
        if normalizer:
            normalizer.load()
        for name, rows in outputs.items():
            texts = [row.text for row in rows]
            layers = {'raw': ' '.join(texts)}
            if normalizer:
                normalized, _, _ = normalizer.normalize(texts)
                layers['sigma'] = ' '.join(normalized)
            results['variants'][name]['scores'] = score_texts(text, layers)
    finally:
        if normalizer:
            normalizer.unload()
            free_gpu()
    (out / 'comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    # Human-readable alternatives, aligned by the same input audio unit, for review without cherry-picking.
    lines = ['# Model comparison', '', results['note'], '']
    lines.extend(['| Variant | Text layer | WER | CER |', '| --- | --- | --- | --- |'])
    for name, result in results['variants'].items():
        for layer in ('raw', 'sigma'):
            score = result['scores'].get(f'{layer}/spec/numbers_to_words')
            if score:
                lines.append(f"| {name} | {layer} | {score['wer']:.2%} | {score['cer']:.2%} |")
    lines.append('')
    for i, unit in enumerate(units):
        lines.append(f'## {unit.id} · {unit.start:.2f}–{unit.end:.2f}s')
        for name in outputs:
            lines.extend(['', f'**{name}:** {outputs[name][i].text}'])
        lines.append('')
    (out / 'alternatives.md').write_text('\n'.join(lines), encoding='utf-8')
    for name, result in results['variants'].items():
        for layer in ('raw', 'sigma'):
            score = result['scores'].get(f'{layer}/spec/numbers_to_words')
            if score:
                print(f"{name:10} {layer:5} WER {score['wer']:.1%}, CER {score['cer']:.1%}")
    print(f'Results: {out / "comparison.json"}')


if __name__ == '__main__':
    main()
