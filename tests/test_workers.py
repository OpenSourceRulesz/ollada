"""Worker env manager tests — no network, no real venvs.

Subprocess calls are captured, not executed, except where a test
explicitly wants a real venv python path check (existence only).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ollada import fit
from ollada.recipes import get_recipe
from ollada.workers import (
    WorkerEnvError,
    _pip_packages,
    ensure_env,
    installed_transformers,
    list_envs,
    rebuild_env,
    venv_path,
    venv_python,
)


def _ok_run(calls):
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_pip_packages_dedupe_and_pin():
    dream = _pip_packages(get_recipe("dream:7b"))
    assert dream == ["pyyaml", "torch", "transformers>=4.53.3,<5",
                     "accelerate", "bitsandbytes"]
    # sdar unions the family's extras: 1.7b alone would omit einops,
    # which 4b needs — the Dell failure. Both models get the same set.
    sdar_17 = _pip_packages(get_recipe("sdar:1.7b"))
    assert sdar_17 == ["pyyaml", "torch", "transformers>=4.57,<5",
                       "accelerate", "einops"]
    sdar_4b = _pip_packages(get_recipe("sdar:4b"))
    assert sdar_4b == sdar_17
    diffcoder = _pip_packages(get_recipe("diffcoder:8b"))
    assert "transformers==5.3.0" in diffcoder


def test_ensure_env_builds_with_gpu_index_on_gpu_machine(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    monkeypatch.setattr(
        fit, "detect_hardware",
        lambda refresh=False: fit.Hardware(gpu_name="Test GPU", vram_total_gb=16.0))
    calls = []
    python = ensure_env(get_recipe("dream:7b"), _run=_ok_run(calls))
    assert python == venv_python("dream")
    assert calls[0][1:3] == ["-m", "venv"]
    pip_calls = [c for c in calls if "pip" in c]
    assert len(pip_calls) == 1  # single install call when a GPU is present
    assert "transformers>=4.53.3,<5" in pip_calls[0]


def test_ensure_env_splits_torch_to_cpu_index_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    monkeypatch.setattr(
        fit, "detect_hardware",
        lambda refresh=False: fit.Hardware(gpu_name=None))
    calls = []
    ensure_env(get_recipe("sdar:4b"), _run=_ok_run(calls))
    pip_calls = [c for c in calls if "pip" in c]
    assert len(pip_calls) == 2
    assert pip_calls[0][pip_calls[0].index("--index-url") + 1] == \
        "https://download.pytorch.org/whl/cpu"
    assert pip_calls[0][-1] == "torch"
    assert "torch" not in pip_calls[1]


def test_ensure_env_missing_package_refuses_with_rebuild_pointer(monkeypatch, tmp_path):
    # The Dell sdar env: right pin, but built from the 1.7b recipe, so
    # einops was missing when 4b ran. That must refuse here, not as a
    # bare ImportError inside the worker.
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    family_dir = venv_path("sdar")
    if sys.platform == "win32":
        (family_dir / "Scripts").mkdir(parents=True)
        (family_dir / "Scripts" / "python.exe").touch()
    else:
        (family_dir / "bin").mkdir(parents=True)
        (family_dir / "bin" / "python").touch()
    monkeypatch.setattr(
        "ollada.workers.installed_transformers", lambda python: "4.57.6")
    monkeypatch.setattr(
        "ollada.workers.installed_distributions",
        lambda python: {"pyyaml", "torch", "transformers", "accelerate"})
    with pytest.raises(WorkerEnvError, match="ollada workers rebuild sdar:4b"):
        ensure_env(get_recipe("sdar:4b"))


def test_ensure_env_refuses_pin_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    family_dir = venv_path("dream")
    if sys.platform == "win32":
        (family_dir / "Scripts").mkdir(parents=True)
        (family_dir / "Scripts" / "python.exe").touch()
    else:
        (family_dir / "bin").mkdir(parents=True)
        (family_dir / "bin" / "python").touch()
    monkeypatch.setattr(
        "ollada.workers.installed_transformers", lambda python: "5.3.0")
    with pytest.raises(WorkerEnvError, match="ollada workers rebuild dream:7b"):
        ensure_env(get_recipe("dream:7b"))


def test_ensure_env_matching_pin_returns_existing_python(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    family_dir = venv_path("dream")
    if sys.platform == "win32":
        (family_dir / "Scripts").mkdir(parents=True)
        (family_dir / "Scripts" / "python.exe").touch()
    else:
        (family_dir / "bin").mkdir(parents=True)
        (family_dir / "bin" / "python").touch()
    monkeypatch.setattr(
        "ollada.workers.installed_transformers", lambda python: "4.57.6")
    assert ensure_env(get_recipe("dream:7b")) == venv_python("dream")


def test_rebuild_deletes_then_builds(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    marker = venv_path("dream") / "marker.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("old")
    monkeypatch.setattr(
        fit, "detect_hardware",
        lambda refresh=False: fit.Hardware(gpu_name="Test GPU", vram_total_gb=16.0))
    calls = []
    rebuild_env(get_recipe("dream:7b"), _run=_ok_run(calls))
    assert not marker.exists()
    assert any("venv" in c for c in calls)


def test_list_envs_reports_built_and_version(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLADA_WORKERS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "ollada.workers.installed_transformers", lambda python: "4.57.6")
    rows = {r["family"]: r for r in list_envs([get_recipe("dream:7b")])}
    assert rows["dream"]["built"] is False
    assert rows["dream"]["transformers"] is None


def test_installed_transformers_returns_none_without_python():
    assert installed_transformers(Path("/nonexistent/python")) is None
