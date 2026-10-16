"""VRAM, timing and GPU-state measurement per pipeline step."""

import gc
import logging
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch

log = logging.getLogger(__name__)

MIB = 1024 * 1024


def gpu_state() -> dict:
    """Driver-level GPU state from nvidia-smi (empty dict if unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return {}
    fields = "name,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,pstate,memory.used,memory.total"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    if not out:
        return {}
    values = [v.strip() for v in out[0].split(",")]
    return dict(zip(fields.split(","), values))


def host_power_state() -> dict:
    state = {}
    for key, path in {
        "ac_online": "/sys/class/power_supply/AC0/online",
        "platform_profile": "/sys/firmware/acpi/platform_profile",
    }.items():
        try:
            with open(path) as f:
                state[key] = f.read().strip()
        except OSError:
            pass
    if "ac_online" not in state:
        import glob

        for p in glob.glob("/sys/class/power_supply/AC*/online"):
            with open(p) as f:
                state["ac_online"] = f.read().strip()
    return state


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def gpu_used_mib() -> float | None:
    state = gpu_state()
    return float(state["memory.used"]) if "memory.used" in state else None


@dataclass
class StepMetrics:
    step: str
    device: str = "cpu"
    load_s: float = 0.0
    process_s: float = 0.0
    audio_s: float | None = None
    rtf: float | None = None  # processing time / audio duration (lower is faster)
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    gpu_used_before_mib: float | None = None
    gpu_used_after_mib: float | None = None
    gpu_state_start: dict = field(default_factory=dict)
    gpu_state_end: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class StepMonitor:
    """Measures one step. Use `with monitor.phase("load"):` and `with monitor.phase("process"):`."""

    def __init__(self, step: str, device: str, audio_s: float | None = None):
        self.m = StepMetrics(step=step, device=device, audio_s=audio_s)
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()

    def __enter__(self) -> "StepMonitor":
        self.m.gpu_state_start = gpu_state()
        self.m.gpu_used_before_mib = gpu_used_mib()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        return self

    @contextmanager
    def phase(self, name: str):
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            setattr(self.m, f"{name}_s", getattr(self.m, f"{name}_s", 0.0) + elapsed)

    def event(self, message: str) -> None:
        log.warning(message)
        self.m.events.append(message)

    def __exit__(self, *exc) -> None:
        if self.cuda:
            self.m.peak_allocated_mib = round(torch.cuda.max_memory_allocated() / MIB, 1)
            self.m.peak_reserved_mib = round(torch.cuda.max_memory_reserved() / MIB, 1)
        if self.m.audio_s:
            self.m.rtf = round(self.m.process_s / self.m.audio_s, 5)
        self.m.load_s = round(self.m.load_s, 3)
        self.m.process_s = round(self.m.process_s, 3)
        self.m.gpu_state_end = gpu_state()

    def after_unload(self, baseline_mib: float | None, tolerance_mib: float = 300) -> None:
        """Record driver-level VRAM after the model was freed and check it is back near baseline."""
        free_gpu()
        self.m.gpu_used_after_mib = gpu_used_mib()
        if baseline_mib is not None and self.m.gpu_used_after_mib is not None:
            excess = self.m.gpu_used_after_mib - baseline_mib
            self.m.extra["vram_returned_to_baseline"] = excess <= tolerance_mib
            if excess > tolerance_mib:
                self.event(f"VRAM {excess:.0f} MiB above baseline after unloading {self.m.step}")
