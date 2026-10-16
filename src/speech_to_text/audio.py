"""Audio decoding and segmentation."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MAX_SEGMENT_S, SAMPLE_RATE, SPLIT_SEARCH_S


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    samples: np.ndarray


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file (m4a, mp3, wav, ...) to mono float32 at `sample_rate`."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH (sudo apt install ffmpeg).")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _quietest_point(audio: np.ndarray, start: int, end: int, window: int) -> int:
    """Index in [start, end) at the centre of the lowest-energy window."""
    region = audio[start:end].astype(np.float64)
    if len(region) <= window:
        return start + len(region) // 2
    energy = np.convolve(region**2, np.ones(window), mode="valid")
    return start + int(np.argmin(energy)) + window // 2


def split_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_segment_s: float = MAX_SEGMENT_S,
    search_s: float = SPLIT_SEARCH_S,
) -> list[Segment]:
    """Split audio into segments of at most `max_segment_s`, cutting at quiet points
    so that words are not chopped in half."""
    max_len = int(max_segment_s * sample_rate)
    search_len = int(search_s * sample_rate)
    window = int(0.1 * sample_rate)  # 100 ms energy window

    segments: list[Segment] = []
    pos, total = 0, len(audio)
    while pos < total:
        if total - pos <= max_len:
            cut = total
        else:
            cut = _quietest_point(audio, pos + max_len - search_len, pos + max_len, window)
        segments.append(Segment(pos / sample_rate, cut / sample_rate, audio[pos:cut]))
        pos = cut
    return segments


def is_silent(samples: np.ndarray, threshold_db: float = -50.0) -> bool:
    """True when the segment's RMS level is below `threshold_db` dBFS."""
    if samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20 * np.log10(rms + 1e-12) < threshold_db
