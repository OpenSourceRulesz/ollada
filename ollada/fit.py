"""Fit check — will this model actually fit on this machine?

The verdict is computed from, in order of trust:
  1. a LOCAL measurement (ollada/measurements.py) — this machine, this GPU;
  2. the recipe's verified peak — ollada's own real-load number, hardware
     labelled (the v0.1 shelf was verified on a Tesla T4);
  3. the recipe's bf16 weights size — labelled an estimate, never a promise.

HEADROOM_GB is reserved on top of the peak for the display driver, the
desktop, and whatever else shares the card. That margin is the DiffCoder
lesson: 15.15GB peak on a 15.6GB T4 leaves ~0.5GB, which is a different
proposition on a card that is also driving monitors.

A recipe may also declare verified.min_host_ram_gb (sdar:4b does — its
load ran out of host RAM on a ~13GB runtime and succeeded on a high-RAM
one; the declared floor is prudent, labelled as observed, not measured).

Detection is dependency-free and failure-tolerant: torch if present,
else nvidia-smi, else "unknown" — and unknown never blocks a load.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

from ollada import measurements
from ollada.recipes import Recipe

HEADROOM_GB = 1.0

FITS = "fits"
TIGHT = "tight"
WONT_FIT = "won't fit"
UNKNOWN = "unknown"


class FitError(RuntimeError):
    """Raised when a load is refused on fit grounds (--force overrides)."""


@dataclass
class Hardware:
    gpu_name: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    ram_total_gb: float | None = None


@dataclass
class FitReport:
    model: str
    verdict: str
    peak_gb: float | None
    peak_source: str
    required_gb: float | None = None
    hardware: Hardware = field(default_factory=Hardware)
    ram_floor_gb: float | None = None
    ram_short: bool = False
    notes: list[str] = field(default_factory=list)

    def blocking_reason(self) -> str | None:
        """Why a load must be refused without --force, or None to proceed."""
        hw = self.hardware
        if self.verdict == WONT_FIT and hw.vram_total_gb is not None and self.peak_gb is not None:
            return (
                f"{self.model} measured a {self.peak_gb:.2f} GB peak ({self.peak_source}); "
                f"this GPU ({hw.gpu_name or 'unknown'}) has {hw.vram_total_gb:.1f} GB total. "
                "Refusing to load — it would OOM mid-load or mid-generation. "
                "Override with --force if you know better."
            )
        if self.ram_short and self.ram_floor_gb is not None and hw.ram_total_gb is not None:
            return (
                f"{self.model} declares a {self.ram_floor_gb:.0f} GB host-RAM floor "
                f"(recipe verified notes); this machine has {hw.ram_total_gb:.1f} GB. "
                "Its load stages weights through host RAM and has failed on less. "
                "Override with --force if you know better."
            )
        return None

    def summary(self) -> str:
        if self.peak_gb is None:
            return f"{self.model}: fit unknown — no peak measurement on record"
        hw = self.hardware
        gpu = hw.gpu_name or "this GPU"
        vram = f"{hw.vram_total_gb:.1f} GB" if hw.vram_total_gb is not None else "unknown VRAM"
        return (
            f"{self.model}: {self.verdict} — peak {self.peak_gb:.2f} GB ({self.peak_source}), "
            f"needs ~{self.required_gb:.1f} GB with headroom; {gpu} has {vram}"
        )


# -- hardware detection ------------------------------------------------------

_cached_hardware: Hardware | None = None


def detect_hardware(refresh: bool = False) -> Hardware:
    global _cached_hardware
    if _cached_hardware is not None and not refresh:
        return _cached_hardware
    hw = Hardware(ram_total_gb=_ram_total_gb())
    _detect_via_torch(hw) or _detect_via_nvidia_smi(hw)
    _cached_hardware = hw
    return hw


def _detect_via_torch(hw: Hardware) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        hw.gpu_name = str(props.name)
        hw.vram_total_gb = props.total_memory / (1024**3)
        try:
            free, _total = torch.cuda.mem_get_info(0)
            hw.vram_free_gb = free / (1024**3)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _detect_via_nvidia_smi(hw: Hardware) -> bool:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            # First invocation under load can be slow; a timeout here
            # used to silently downgrade a GPU machine to CPU torch.
            timeout=15,
        )
        if out.returncode != 0:
            return False
        # Take the first line that actually parses — a warning line
        # ahead of the data must not poison the whole detection.
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                name, total_mib, free_mib = parts
                hw.vram_total_gb = float(total_mib) / 1024
                hw.vram_free_gb = float(free_mib) / 1024
            except ValueError:
                continue
            hw.gpu_name = name
            return True
        return False
    except Exception:
        return False


def _ram_total_gb() -> float | None:
    try:  # POSIX (Linux/macOS)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows
        import ctypes

        class _MemStat(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStat()
        stat.dwLength = ctypes.sizeof(_MemStat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / (1024**3)
    except Exception:
        pass
    return None


# -- the check ----------------------------------------------------------------


def peak_for(recipe: Recipe, record: dict | None = None) -> tuple[float | None, str]:
    """(peak_gb, source label) by trust order: local, verified, estimate."""
    record = record if record is not None else measurements.load_record(recipe.name)
    if record and record.get("peak_vram_gb"):
        where = record.get("hardware") or "this machine"
        return float(record["peak_vram_gb"]), f"measured on this machine ({where})"
    if recipe.verified.get("vram_gb"):
        where = recipe.verified.get("hardware") or "verified hardware"
        return float(recipe.verified["vram_gb"]), f"ollada-verified on {where}"
    if recipe.size_gb_bf16:
        return float(recipe.size_gb_bf16), "weights-size estimate (unverified)"
    return None, "no data"


def assess(
    recipe: Recipe,
    hardware: Hardware | None = None,
    record: dict | None = None,
) -> FitReport:
    hw = hardware if hardware is not None else detect_hardware()
    peak, source = peak_for(recipe, record)
    floor = recipe.verified.get("min_host_ram_gb")
    ram_floor = float(floor) if floor else None
    ram_short = bool(
        ram_floor is not None and hw.ram_total_gb is not None and hw.ram_total_gb < ram_floor
    )
    report = FitReport(
        model=recipe.name,
        verdict=UNKNOWN,
        peak_gb=peak,
        peak_source=source,
        required_gb=round(peak + HEADROOM_GB, 2) if peak is not None else None,
        hardware=hw,
        ram_floor_gb=ram_floor,
        ram_short=ram_short,
    )
    if ram_short:
        report.notes.append(
            f"host RAM {hw.ram_total_gb:.1f} GB is below this recipe's {ram_floor:.0f} GB floor"
        )
    if peak is None or hw.vram_total_gb is None:
        if hw.vram_total_gb is None:
            report.notes.append("no GPU detected — VRAM verdict unavailable")
        return report
    if peak > hw.vram_total_gb:
        report.verdict = WONT_FIT
    elif report.required_gb > hw.vram_total_gb:
        report.verdict = TIGHT
        report.notes.append("fits only with the card otherwise idle")
    elif hw.vram_free_gb is not None and hw.vram_free_gb < report.required_gb:
        report.verdict = TIGHT
        report.notes.append(f"fits the card, but only {hw.vram_free_gb:.1f} GB is free right now")
    else:
        report.verdict = FITS
    return report
