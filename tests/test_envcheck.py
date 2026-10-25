"""Transformers-pin enforcement tests.

The 2026-10-25 desktop run proved the failure mode: a fresh [gpu] install
lands transformers 5.x, and Dream's custom code then crashes deep in load
with a RoPE 'default' KeyError. The manager must refuse before load with
the exact pip line, not after with a traceback.
"""

import pytest

from ollada.recipes import env_mismatch, spec_satisfied


def test_spec_operators():
    assert spec_satisfied("==4.53.3", "4.53.3") is True
    assert spec_satisfied("==4.53.3", "4.53.4") is False
    assert spec_satisfied(">=4.57", "4.57.6") is True
    assert spec_satisfied(">=4.57", "4.53.3") is False
    assert spec_satisfied(">=4.53.3,<5", "4.57.6") is True
    assert spec_satisfied(">=4.53.3,<5", "5.0.0") is False
    assert spec_satisfied(">=4.53.3,<5", "4.53.3") is True
    assert spec_satisfied("==5.3.0", "5.3.0") is True
    assert spec_satisfied("==5.3.0", "5.3.1") is False
    assert spec_satisfied("4.53.3", "4.53.3") is True  # bare version reads as ==
    assert spec_satisfied("!=4.53.3", "4.57.6") is True
    assert spec_satisfied("!=4.53.3", "4.53.3") is False


def test_spec_unparseable_fails_closed():
    assert spec_satisfied("banana", "4.53.3") is False
    assert spec_satisfied(">=banana", "4.53.3") is False


def test_env_mismatch_states():
    assert env_mismatch({}) is None  # no pin, nothing to check
    assert env_mismatch({"transformers": ">=4.53.3,<5"}, "4.57.6") is None
    assert env_mismatch({"transformers": "==5.3.0"}, "4.57.6") is not None
    msg = env_mismatch({"transformers": ">=4.53.3,<5"}, "5.3.0")
    assert msg is not None
    assert "5.3.0" in msg and "transformers>=4.53.3,<5" in msg
    assert 'pip install "transformers>=4.53.3,<5"' in msg


def test_manager_refuses_on_pin_mismatch_before_creating_adapter(monkeypatch):
    import ollada.runtime.manager as manager_mod
    from ollada import fit
    from ollada.recipes import get_recipe

    created = []
    monkeypatch.setattr(
        manager_mod, "create_adapter", lambda r: created.append(r) or _FakeAdapter(r)
    )
    # Pin the env check to "installed 5.x" regardless of this machine.
    monkeypatch.setattr(
        manager_mod, "env_mismatch",
        lambda env: "mismatch" if env.get("transformers") else None,
    )
    manager = manager_mod.ModelManager()
    with pytest.raises(fit.FitError):
        manager.ensure_loaded(get_recipe("dream:7b"))
    assert created == []  # refused before any adapter existed
    adapter = manager.ensure_loaded(get_recipe("dream:7b"), force=True)
    assert adapter.loaded is True


class _FakeAdapter:
    def __init__(self, recipe):
        self.recipe = recipe
        self.loaded = True

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False
