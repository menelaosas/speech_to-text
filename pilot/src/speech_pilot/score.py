"""Score a run against the operator's reference excerpt (WER/CER, optional DER)."""

import json
import re
from pathlib import Path

import jiwer

from .greek_text import normalize
from .schema import Transcript

DER_COLLAR_S = 0.25


def _reference_text(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"^\s*[^:\s]{1,20}:\s*", "", line)  # drop "LABEL:" turn prefixes
        line = line.replace("[...]", " ")
        lines.append(line)
    return " ".join(lines)


def _rttm_annotation(path: Path, uri: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            ann[Segment(start, start + dur)] = p[7]
    return ann


def score_run(run_dir: Path) -> dict:
    ref_dir = run_dir / "reference"
    ref_txt = ref_dir / "excerpt.txt"
    if not ref_txt.exists():
        raise FileNotFoundError(f"{ref_txt} not found: the operator has not written the reference yet")
    excerpt = json.loads((ref_dir / "excerpt.json").read_text(encoding="utf-8"))
    t = Transcript.model_validate_json((run_dir / "transcript.json").read_text(encoding="utf-8"))
    window = [s for s in t.segments if s.id in set(excerpt["segments"])]
    reference = _reference_text(ref_txt)

    results: dict = {"excerpt": {k: excerpt[k] for k in ("start_s", "end_s", "speakers")},
                     "normalizer": "lowercase, strip tonos/dialytika, ς->σ, strip punctuation, "
                                   "unify εφτα/οχτω/εννια, (numbers to words)"}
    layers = {"raw": " ".join(s.text for s in window)}
    if all(s.text_normalized is not None for s in window):
        layers["sigma_normalized"] = " ".join(s.text_normalized for s in window)
    for layer, hyp in layers.items():
        for numbers in (True, False):
            r, h = normalize(reference, numbers), normalize(hyp, numbers)
            key = f"{layer}{'' if numbers else '_no_number_normalization'}"
            results[key] = {"wer": round(jiwer.wer(r, h), 4), "cer": round(jiwer.cer(r, h), 4),
                            "ref_words": len(r.split()), "hyp_words": len(h.split())}

    ref_rttm = ref_dir / "excerpt.rttm"
    if ref_rttm.exists():
        from pyannote.core import Segment as PSegment
        from pyannote.metrics.diarization import DiarizationErrorRate

        hyp = _rttm_annotation(run_dir / "diarization.rttm", "hyp")
        start, end = excerpt["start_s"], excerpt["end_s"]
        hyp = _shift(hyp, start, end)  # reference times are relative to the start of excerpt.wav
        ref = _rttm_annotation(ref_rttm, "hyp")
        metric = DiarizationErrorRate(collar=DER_COLLAR_S, skip_overlap=False)
        der = metric(ref, hyp, detailed=True, uem=PSegment(0, end - start))
        results["der"] = {"collar_s": DER_COLLAR_S, "skip_overlap": False,
                          **{k: round(float(v), 4) for k, v in der.items()}}
    (ref_dir / "score.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _shift(ann, start: float, end: float):
    from pyannote.core import Annotation, Segment

    out = Annotation(uri=ann.uri)
    for seg, track, label in ann.crop(Segment(start, end)).itertracks(yield_label=True):
        out[Segment(seg.start - start, seg.end - start), track] = label
    return out
