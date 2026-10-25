"""Manager worker-path tests.

The happy path through a real worker subprocess with a real model needs
torch and a GPU, so it is verified on hardware (PLAN.md section 7), not
here. What these prove without hardware:
- a fit refusal happens before any env build or worker spawn;
- a real worker subprocess boots, answers, and reports an adapter
  failure as an AdapterError (not a hang, not a stray traceback);
- the in-process escape hatch still works for tests and debugging.
"""

import sys

import pytest

from ollada import fit
from ollada.adapters.base import AdapterError
from ollada.recipes import get_recipe
import ollada.runtime.manager as manager_mod
from ollada.runtime.manager import ModelManager


class _FakeAdapter:
    def __init__(self, recipe):
        self.recipe = recipe
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat(self, messages, overrides=None):
        from ollada.adapters.base import GenResult
        return GenResult(text="in-process", tokens=1, seconds=0.01)


def _capable_hardware(monkeypatch):
    monkeypatch.setattr(
        fit, "detect_hardware",
        lambda refresh=False: fit.Hardware(
            gpu_name="Test GPU", vram_total_gb=24.0, vram_free_gb=23.0, ram_total_gb=64.0))


def test_worker_path_refuses_before_build_or_spawn(monkeypatch):
    monkeypatch.setattr(
        fit, "detect_hardware",
        lambda refresh=False: fit.Hardware(
            gpu_name="Tiny GPU", vram_total_gb=8.0, vram_free_gb=7.0, ram_total_gb=32.0))
    calls = []
    monkeypatch.setattr(manager_mod.workers, "ensure_env",
                        lambda recipe: calls.append("ensure_env"))
    monkeypatch.setattr(manager_mod.WorkerAdapter, "_start",
                        lambda self: calls.append("spawn"))
    with pytest.raises(fit.FitError):
        ModelManager().ensure_loaded(get_recipe("dream:7b"))  # 9.80 GB peak, 8 GB card
    assert calls == []  # refused before any env build or worker spawn


def test_real_worker_subprocess_reports_adapter_failure(monkeypatch):
    _capable_hardware(monkeypatch)
    # Point the "worker env python" at this interpreter: the real worker
    # boots, then dream's adapter import fails (no torch in the sandbox)
    # and the failure must arrive as an AdapterError, not a hang.
    monkeypatch.setattr(manager_mod.workers, "ensure_env", lambda recipe: sys.executable)
    manager = ModelManager()
    with pytest.raises(AdapterError):
        manager.ensure_loaded(get_recipe("dream:7b"))
    assert manager.status() == {"loaded": None, "family": None}
    assert manager_mod._LIVE == set()  # the failed worker was reaped, not orphaned


def test_inprocess_escape_hatch_still_works(monkeypatch):
    _capable_hardware(monkeypatch)
    monkeypatch.setenv("OLLADA_INPROCESS", "1")
    monkeypatch.setattr(manager_mod, "create_adapter", lambda recipe: _FakeAdapter(recipe))
    manager = ModelManager()
    adapter = manager.ensure_loaded(get_recipe("dream:7b"))
    assert adapter.loaded is True
    assert manager.chat("dream:7b", [{"role": "user", "content": "hi"}]).text == "in-process"
