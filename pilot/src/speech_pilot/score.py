"""Score a run against the operator's reference excerpt (WER/CER, optional DER)."""

import json
import re
from pathlib import Path

import jiwer

from .greek_text import normalize, normalize_lenient
from .schema import Transcript

DER_COLLAR_S = 0.25
WORDS_FILE = Path(__file__).resolve().parent / "resources" / "el_words.txt"


def import_numbered_reference(src: Path, dst: Path) -> dict:
    """Convert an operator transcript written as `<number> <text>` per line into `LABEL: text`.

    Labels of 2+ digits whose digits are all known single-speaker labels (e.g. "12", "32") are read as
    overlapping speech of those speakers ("1+2"). Lines without a number get "?". The source file is
    only read."""
    lines = [l.rstrip() for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    parsed = []
    for line in lines:
        m = re.match(r"^\s*(\d+)\s+(.*)$", line)
        parsed.append((m.group(1), m.group(2)) if m else (None, line.strip()))
    singles = {lab for lab, _ in parsed if lab and len(lab) == 1}
    out, overlap_labels = [], {}
    for lab, text in parsed:
        if lab is None:
            label = "?"
        elif len(lab) > 1 and set(lab) <= singles:
            label = "+".join(lab)
            overlap_labels[lab] = label
        else:
            label = lab
        out.append(f"{label}: {text}")
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    speakers = sorted(singles, key=int)
    return {"source": src.name, "lines": len(lines), "speakers": speakers,
            "overlap_labels": overlap_labels, "unlabelled_lines": sum(1 for lab, _ in parsed if lab is None),
            "rule": "N text -> N: text; multi-digit labels made of known single labels = overlap (12 -> 1+2); "
                    "no number -> ?"}


def parse_reference(path: Path) -> list[tuple[str, str]]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([^:\s]{1,20}):\s*(.*)$", line)
        if m:
            turns.append((m.group(1), m.group(2).replace("[...]", " ")))
        elif line.strip():
            turns.append(("?", line.strip()))
    return turns


def _non_words(text: str) -> list[str]:
    """Reference words (after normalisation) that are not in the Greek word list: typing slips, cut-off
    words or rare names. Reported, never corrected."""
    known = {normalize(w, numbers=False) for w in WORDS_FILE.read_text(encoding="utf-8").split()}
    return [w for w in normalize_lenient(text, numbers=True).split() if w not in known]


def align_reference(ref_dir: Path, cfg) -> dict:
    """Build `excerpt.rttm` from the operator's transcript: speaker identity comes from the operator's
    labels, the start/end of each line from Greek CTC forced alignment of the operator's own words to
    `excerpt.wav`. Overlap lines ("1+2") give a turn to each listed speaker over the same span; lines
    labelled "?" are left out. The result is marked as derived in `excerpt_alignment.json`."""
    from . import audio as A
    from .providers.ctc_aligner import CtcAligner

    turns = parse_reference(ref_dir / "excerpt.txt")
    wav16 = ref_dir / "excerpt_16k.wav"
    cmd = A.prepare_wav(ref_dir / "excerpt.wav", wav16, 16000)
    audio, sr = A.read_wav(wav16)
    tokens, line_of = [], []
    for i, (_, text) in enumerate(turns):
        for tok in text.split():
            tokens.append(tok)
            line_of.append(i)
    aligner = CtcAligner(cfg.alignment, cfg.device.name)
    aligner.load()
    times = aligner.align(audio, sr, tokens)
    info = aligner.info()
    aligner.unload()

    lines = []
    for i, (label, text) in enumerate(turns):
        ws = [t for t, li in zip(times, line_of) if li == i and t.aligned]
        lines.append({"line": i + 1, "label": label, "text": text,
                      "start": round(min(w.start for w in ws), 3) if ws else None,
                      "end": round(max(w.end for w in ws), 3) if ws else None,
                      "mean_score": round(sum(w.score for w in ws) / len(ws), 3) if ws else None})
    rttm = []
    for ln in lines:
        if ln["start"] is None or ln["label"] == "?":
            continue
        for spk in ln["label"].split("+"):
            rttm.append(f"SPEAKER excerpt 1 {ln['start']:.3f} {ln['end'] - ln['start']:.3f} <NA> <NA> {spk} <NA> <NA>")
    (ref_dir / "excerpt.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")
    meta = {"derived": "speaker labels: operator; times: forced alignment of the operator's text (not hand-marked)",
            "aligner": info.model_dump(), "audio_prep": cmd, "lines": lines,
            "words": [{"text": tok, "line": li + 1, "start": round(t.start, 3), "end": round(t.end, 3),
                       "score": t.score, "aligned": t.aligned} for tok, li, t in zip(tokens, line_of, times)],
            "skipped_lines": [ln["line"] for ln in lines if ln["start"] is None or ln["label"] == "?"]}
    (ref_dir / "excerpt_alignment.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"rttm_turns": len(rttm), "lines": len(lines), "skipped_lines": meta["skipped_lines"],
            "mean_word_score": round(sum(t.score for t in times) / max(len(times), 1), 3)}


def _word_errors(ref: str, hyp: str) -> int:
    r, h = ref.split(), hyp.split()
    if not r:
        return len(h)
    if not h:
        return len(r)
    out = jiwer.process_words(" ".join(r), " ".join(h))
    return out.substitutions + out.deletions + out.insertions


def cp_wer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> dict:
    """Concatenated minimum-permutation WER: text is concatenated per speaker on both sides, speakers are
    matched one-to-one to minimise errors (unmatched speakers count all their words as errors)."""
    from scipy.optimize import linear_sum_assignment

    refs = {k: normalize(v) for k, v in ref_by_speaker.items()}
    hyps = {k: normalize(v) for k, v in hyp_by_speaker.items()}
    rk, hk = list(refs), list(hyps)
    n = max(len(rk), len(hk))
    cost = [[_word_errors(refs[rk[i]] if i < len(rk) else "", hyps[hk[j]] if j < len(hk) else "")
             for j in range(n)] for i in range(n)]
    rows, cols = linear_sum_assignment(cost)
    total_ref = sum(len(v.split()) for v in refs.values())
    errors = sum(cost[i][j] for i, j in zip(rows, cols))
    mapping = {rk[i]: (hk[j] if j < len(hk) else None) for i, j in zip(rows, cols) if i < len(rk)}
    return {"cpwer": round(errors / max(total_ref, 1), 4), "errors": int(errors), "ref_words": total_ref,
            "mapping": mapping, "unmatched_hyp_speakers": [hk[j] for i, j in zip(rows, cols)
                                                           if i >= len(rk) and j < len(hk)]}


def hyp_by_speaker(t: Transcript, start: float, end: float, segment_ids: set[str] | None = None) -> dict[str, str]:
    """Hypothesis text per speaker inside the window. Word-timed transcripts are cut at word level
    (word midpoint inside the window); otherwise whole segments are used."""
    out: dict[str, list[str]] = {}
    for s in t.segments:
        spk = s.primary_speaker or "UNK"
        if s.words:
            toks = [(w.text_normalized or w.text) for w in s.words if start <= (w.start + w.end) / 2 < end]
        elif (segment_ids and s.id in segment_ids) or (not segment_ids and start <= (s.start + s.end) / 2 < end):
            toks = [s.text_normalized or s.text]
        else:
            toks = []
        if toks:
            out.setdefault(spk, []).extend(toks)
    return {k: " ".join(v) for k, v in out.items()}


def _rttm_annotation(path: Path, uri: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for i, line in enumerate(path.read_text().splitlines()):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            # a unique track per line: two speakers on the same span must not overwrite each other
            ann[Segment(start, start + dur), f"t{i}"] = p[7]
    return ann


def score_texts(reference: str, layers: dict[str, str]) -> dict:
    results = {}
    for layer, hyp in layers.items():
        for norm_name, fn in (("spec", normalize), ("lenient", normalize_lenient)):
            for numbers in (True, False):
                r, h = fn(reference, numbers), fn(hyp, numbers)
                key = f"{layer}/{norm_name}/{'numbers_to_words' if numbers else 'numbers_as_written'}"
                out = jiwer.process_words(r, h)
                results[key] = {"wer": round(out.wer, 4), "cer": round(jiwer.cer(r, h), 4),
                                "substitutions": out.substitutions, "deletions": out.deletions,
                                "insertions": out.insertions, "ref_words": len(r.split()),
                                "hyp_words": len(h.split())}
    return results


def score_run(run_dir: Path, ref_dir: Path | None = None) -> dict:
    """Score `run_dir` against a reference folder (default: the run's own `reference/`)."""
    own_reference = ref_dir is None or Path(ref_dir).resolve() == (run_dir / "reference").resolve()
    ref_dir = Path(ref_dir) if ref_dir else run_dir / "reference"
    ref_txt = ref_dir / "excerpt.txt"
    if not ref_txt.exists():
        raise FileNotFoundError(f"{ref_txt} not found: the operator has not written the reference yet")
    excerpt = json.loads((ref_dir / "excerpt.json").read_text(encoding="utf-8"))
    t = Transcript.model_validate_json((run_dir / "transcript.json").read_text(encoding="utf-8"))
    start, end = excerpt["start_s"], excerpt["end_s"]
    turns = parse_reference(ref_txt)
    reference = " ".join(text for _, text in turns)

    # The excerpt's segment ids are only meaningful for the run that chose the excerpt.
    if excerpt.get("source_run", run_dir.name) == run_dir.name and not any(s.words for s in t.segments):
        window = [s for s in t.segments if s.id in set(excerpt["segments"])]
        seg_ids = set(excerpt["segments"])
    else:  # another run's segmentation: cut at word level when word times exist, else by segment midpoint
        window = [s for s in t.segments if start <= (s.start + s.end) / 2 < end]
        seg_ids = None
    if any(s.words for s in t.segments):
        words = [w for s in t.segments for w in (s.words or []) if start <= (w.start + w.end) / 2 < end]
        layers = {"raw": " ".join(w.text for w in words)}
        if all(w.text_normalized is not None for w in words):
            layers["sigma_normalized"] = " ".join(w.text_normalized for w in words)
    else:
        layers = {"raw": " ".join(s.text for s in window)}
        if all(s.text_normalized is not None for s in window):
            layers["sigma_normalized"] = " ".join(s.text_normalized for s in window)

    ref_speakers = sorted({p for lab, _ in turns if lab != "?" for p in lab.split("+")})
    hyp_primary = sorted({s.primary_speaker for s in window if s.primary_speaker})
    hyp_any = sorted({sp for s in window for sp in s.speakers})
    non_words = _non_words(reference)
    results: dict = {
        "excerpt": {k: excerpt[k] for k in ("start_s", "end_s")},
        "normalizers": {
            "spec": "lowercase, strip tonos/dialytika, ς->σ, numbers to words (optional), strip punctuation, "
                    "unify εφτα/οχτω/εννια",
            "lenient": "spec + letters held 3+ times collapsed to one + hesitations removed "
                       f"({', '.join(sorted(__import__('speech_pilot.greek_text', fromlist=['FILLERS']).FILLERS))})",
        },
        "speakers": {"reference": ref_speakers, "reference_count": len(ref_speakers),
                     "reference_overlap_turns": sum(1 for lab, _ in turns if "+" in lab),
                     "diarization_primary": hyp_primary, "diarization_any": hyp_any},
        "reference_quality": {"words": len(normalize_lenient(reference).split()),
                              "not_in_greek_word_list": len(non_words), "examples": non_words[:40]},
        "scores": score_texts(reference, layers),
    }

    # Speaker-attributed WER: reference text per operator speaker ("1+2" lines count for the first listed).
    ref_spk: dict[str, list[str]] = {}
    for label, text in turns:
        ref_spk.setdefault(label.split("+")[0], []).append(text)
    results["cpwer"] = cp_wer({k: " ".join(v) for k, v in ref_spk.items()}, hyp_by_speaker(t, start, end, seg_ids))
    results["cpwer"]["note"] = ("spec normaliser on the ς-normalised layer when available; operator overlap lines "
                                "count for the first listed speaker")

    ref_rttm = ref_dir / "excerpt.rttm"
    if ref_rttm.exists():
        from pyannote.core import Segment as PSegment
        from pyannote.metrics.diarization import DiarizationErrorRate

        hyp = _shift(_rttm_annotation(run_dir / "diarization.rttm", "hyp"), start, end)
        ref = _rttm_annotation(ref_rttm, "hyp")
        results["der"] = {"reference": "operator speaker labels + forced-alignment times (excerpt_alignment.json)"}
        for collar in (DER_COLLAR_S, 0.0):
            for skip_overlap in (False, True):
                metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
                der = metric(ref, hyp, detailed=True, uem=PSegment(0, end - start))
                total = der["total"]
                results["der"][f"collar_{collar}_skip_overlap_{skip_overlap}"] = {
                    "der": round(float(der["diarization error rate"]), 4),
                    "missed": round(der["missed detection"] / total, 4),
                    "false_alarm": round(der["false alarm"] / total, 4),
                    "confusion": round(der["confusion"] / total, 4), "scored_s": round(float(total), 2)}
    else:
        results["der"] = "not computed: no reference RTTM (speaker timings); run `speech-pilot align-reference`"
    results["run"] = run_dir.name
    out_name = "score.json" if own_reference else f"score_{run_dir.name}.json"
    (ref_dir / out_name).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _shift(ann, start: float, end: float):
    from pyannote.core import Annotation, Segment

    out = Annotation(uri=ann.uri)
    for seg, track, label in ann.crop(Segment(start, end)).itertracks(yield_label=True):
        out[Segment(seg.start - start, seg.end - start), track] = label
    return out
