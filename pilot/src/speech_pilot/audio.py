"""Audio inspection and preparation. The original file is only ever read."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"{tool} is not installed or not on PATH")
    return exe


def ffprobe(path: Path) -> dict:
    cmd = [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def _ffmpeg_filter_report(path: Path, af: str) -> str:
    cmd = [_require("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-af", af, "-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stderr


def measure_loudness(path: Path) -> dict:
    """EBU R128 integrated loudness / LRA / true peak, plus per-channel level and clipping stats."""
    ebu = _ffmpeg_filter_report(path, "ebur128=peak=true")
    summary = ebu[ebu.rfind("Summary:"):]

    def grab(label: str) -> float | None:
        m = re.search(label + r":\s+(-?[\d.]+|-inf)", summary)
        return None if not m or m.group(1) == "-inf" else float(m.group(1))

    stats = _ffmpeg_filter_report(path, "astats=measure_overall=none:measure_perchannel=all")
    channels: list[dict] = []
    for block in re.split(r"\] Channel: \d+", stats)[1:]:
        ch = {}
        for key, label in {
            "rms_db": "RMS level dB",
            "peak_db": "Peak level dB",
            "noise_floor_db": "Noise floor dB",
            "peak_count": "Peak count",  # how often the peak level is reached (not a clipping count)
            "dc_offset": "DC offset",
        }.items():
            m = re.search(rf"(?:{label}):\s+(-?[\d.e+-]+|-inf)", block)
            if m:
                ch[key] = None if m.group(1) == "-inf" else float(m.group(1))
        channels.append(ch)
    return {
        "integrated_lufs": grab("I"),
        "loudness_range_lu": grab("LRA"),
        "true_peak_dbfs": grab("Peak"),
        "channels": channels,
    }


def count_clipping(path: Path, threshold: float = 0.999, chunk_samples: int = 1 << 20) -> dict:
    """Decode the file at its own sample rate (streamed) and count samples at or near full scale,
    plus runs of 3+ consecutive full-scale samples (a stronger sign of real clipping)."""
    cmd = [_require("ffmpeg"), "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    total = clipped = runs = 0
    prev_tail = 0
    while block := proc.stdout.read(chunk_samples * 4):
        x = np.abs(np.frombuffer(block[: len(block) // 4 * 4], dtype=np.float32)) >= threshold
        total += x.size
        clipped += int(x.sum())
        if x.any():
            padded = np.concatenate([[False], x, [False]]).astype(np.int8)
            starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
            lengths = ends - starts
            if prev_tail and len(starts) and starts[0] == 0:
                lengths[0] += prev_tail
            runs += int((lengths >= 3).sum())
            prev_tail = int(lengths[-1]) if len(ends) and ends[-1] == x.size else 0
        else:
            prev_tail = 0
    proc.wait()
    return {"threshold": threshold, "samples": total, "near_full_scale_samples": clipped,
            "fraction": clipped / total if total else 0.0, "runs_of_3_or_more": runs}


def prepare_wav(src: Path, dst: Path, sample_rate: int, channel: int | None = None,
                excerpt: tuple[float, float] | None = None) -> list[str]:
    """Write a 16-bit PCM WAV processing copy. Returns the exact ffmpeg command used."""
    cmd = [_require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if excerpt:
        cmd += ["-ss", f"{excerpt[0]:.3f}", "-to", f"{excerpt[1]:.3f}"]
    cmd += ["-i", str(src)]
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)]
    subprocess.run(cmd, check=True)
    return cmd


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def export_excerpt(src: Path, dst: Path, start: float, end: float) -> list[str]:
    """Listening copy of an excerpt at the original sample rate (WAV, lossless from the decoded source)."""
    cmd = [_require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return cmd


def quietest_point(audio: np.ndarray, start: int, end: int, window: int) -> int:
    """Sample index in [start, end) at the centre of the lowest-energy window."""
    region = audio[start:end].astype(np.float64)
    if len(region) <= window:
        return start + len(region) // 2
    energy = np.convolve(region**2, np.ones(window), mode="valid")
    return start + int(np.argmin(energy)) + window // 2


def split_long(audio: np.ndarray, sr: int, start_s: float, end_s: float,
               max_s: float, search_s: float) -> list[tuple[float, float]]:
    """Split [start_s, end_s] into pieces of at most `max_s`, cutting at quiet points."""
    pieces = []
    pos, end = int(start_s * sr), int(end_s * sr)
    max_len, search, window = int(max_s * sr), int(search_s * sr), int(0.1 * sr)
    while pos < end:
        cut = end if end - pos <= max_len else quietest_point(audio, pos + max_len - search, pos + max_len, window)
        pieces.append((pos / sr, cut / sr))
        pos = cut
    return pieces
