"""Local measurement records — what each model actually took on THIS machine.

Borrowed shape (not code) from OllamaDiffuser's memory_record.py, analysed
2026-10-22: every load and generation records its peak VRAM, the highest
seen is kept per model, and the fit check (ollada/fit.py) prefers a local
measurement over the recipe's verified number once one exists.

Differences from theirs, on purpose:
- Records live in ~/.ollada/measurements/ and NEVER leave the machine
  (no telemetry, no upload — ollada's standing $0/no-phone-home rule).
- One model loads at a time (and, later, one model per worker process),
  so a process peak IS the model's peak — no share-subtraction math.

A record: {"model", "peak_vram_gb", "observations", "hardware", "updated"}.
Peaks below _MIN_PLAUSIBLE_GB are ignored: a model that never touched the
GPU (CPU fallback, failed load) must not write a flattering fake number.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_MIN_PLAUSIBLE_GB = 0.05


def measurements_dir() -> Path:
    override = os.environ.get("OLLADA_MEASUREMENTS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".ollada" / "measurements"


def _path(model: str, directory: Path | None = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return (directory or measurements_dir()) / f"{safe}.json"


def load_record(model: str, directory: Path | None = None) -> dict | None:
    """The local measurement record for a model, or None if never measured here."""
    try:
        with _path(model, directory).open("r", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or "peak_vram_gb" not in record:
        return None
    return record


def all_records(directory: Path | None = None) -> dict[str, dict]:
    folder = directory or measurements_dir()
    records: dict[str, dict] = {}
    if not folder.is_dir():
        return records
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and "model" in record and "peak_vram_gb" in record:
            records[record["model"]] = record
    return records


def record_peak(
    model: str,
    peak_vram_gb: float,
    hardware: str | None = None,
    directory: Path | None = None,
) -> dict | None:
    """Merge one observed peak into the model's record (max wins)."""
    if peak_vram_gb is None or peak_vram_gb < _MIN_PLAUSIBLE_GB:
        return load_record(model, directory)
    existing = load_record(model, directory)
    record = {
        "model": model,
        "peak_vram_gb": max(float(peak_vram_gb), float(existing["peak_vram_gb"]) if existing else 0.0),
        "observations": int(existing["observations"]) + 1 if existing else 1,
        "hardware": hardware or (existing or {}).get("hardware"),
        "updated": time.strftime("%Y-%m-%d"),
    }
    folder = directory or measurements_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = _path(model, directory).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        tmp.replace(_path(model, directory))
    except OSError:
        return existing  # measurements are advisory; never break a run over them
    return record


# -- CUDA peak capture (lazy torch; a no-op without a GPU) -------------------


def _torch_cuda():
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def gpu_name() -> str | None:
    torch = _torch_cuda()
    if torch is None:
        return None
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def reset_cuda_peak() -> None:
    torch = _torch_cuda()
    if torch is not None:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def cuda_peak_gb() -> float | None:
    torch = _torch_cuda()
    if torch is None:
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return None


def capture_and_record(model: str, directory: Path | None = None) -> dict | None:
    """Record the CUDA peak since the last reset, then reset for the next run."""
    peak = cuda_peak_gb()
    if peak is None:
        return None
    record = record_peak(model, peak, hardware=gpu_name(), directory=directory)
    reset_cuda_peak()
    return record
