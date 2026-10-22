"""Fit-check tests — verdict matrix, peak trust order, and the refusal
chain (manager raises before creating an adapter; the API maps it to 409;
options.force overrides). Hardware is fabricated; no GPU involved."""

import pytest

from ollada import fit
from ollada.recipes import Recipe, get_recipe


def make_recipe(**kw) -> Recipe:
    base = dict(
        name="fake:1b",
        family="fake",
        source="org/fake",
        license="MIT",
        task="chat",
        description="",
        size_gb_bf16=2.0,
        quant_default="bf16",
        dtype="float16",
        generate={"steps": 1},
        verified={},
    )
    base.update(kw)
    return Recipe(**base)


def hw(**kw) -> fit.Hardware:
    base = dict(gpu_name="Test GPU", vram_total_gb=16.0, vram_free_gb=15.0, ram_total_gb=32.0)
    base.update(kw)
    return fit.Hardware(**base)


# -- peak trust order ---------------------------------------------------------


def test_local_measurement_beats_recipe_verified():
    recipe = get_recipe("dream:7b")  # verified 9.80 GB on a T4
    record = {"peak_vram_gb": 10.4, "hardware": "RTX Test", "observations": 3}
    peak, source = fit.peak_for(recipe, record)
    assert peak == 10.4
    assert "this machine" in source


def test_recipe_verified_is_second():
    peak, source = fit.peak_for(get_recipe("sdar:4b"), None)
    assert peak == 8.87
    assert "Tesla T4" in source


def test_unverified_falls_back_to_labelled_estimate():
    peak, source = fit.peak_for(make_recipe(), None)
    assert peak == 2.0
    assert "estimate" in source and "unverified" in source


# -- verdict matrix -------------------------------------------------------------


def test_fits_with_headroom():
    report = fit.assess(make_recipe(verified={"vram_gb": 9.8}), hardware=hw())
    assert report.verdict == fit.FITS
    assert report.blocking_reason() is None


def test_tight_when_only_the_headroom_overflows():
    # the DiffCoder-on-T4 case: 15.15 GB peak on a 15.6 GB card
    report = fit.assess(
        make_recipe(verified={"vram_gb": 15.15}), hardware=hw(vram_total_gb=15.6, vram_free_gb=15.0)
    )
    assert report.verdict == fit.TIGHT
    assert report.blocking_reason() is None  # tight warns; it does not refuse


def test_wont_fit_and_blocks():
    report = fit.assess(make_recipe(verified={"vram_gb": 15.15}), hardware=hw(vram_total_gb=12.0))
    assert report.verdict == fit.WONT_FIT
    reason = report.blocking_reason()
    assert reason and "--force" in reason


def test_tight_when_free_vram_is_low():
    report = fit.assess(
        make_recipe(verified={"vram_gb": 9.8}),
        hardware=hw(vram_total_gb=24.0, vram_free_gb=10.0),
    )
    assert report.verdict == fit.TIGHT


def test_unknown_without_gpu_never_blocks():
    report = fit.assess(
        make_recipe(verified={"vram_gb": 99.0}), hardware=fit.Hardware(ram_total_gb=32.0)
    )
    assert report.verdict == fit.UNKNOWN
    assert report.blocking_reason() is None


def test_ram_floor_blocks_only_when_declared():
    sdar = get_recipe("sdar:4b")  # declares min_host_ram_gb in its verified block
    report = fit.assess(sdar, hardware=hw(ram_total_gb=8.0))
    assert report.ram_short is True
    assert report.blocking_reason() is not None
    dream = get_recipe("dream:7b")  # no floor declared -> no RAM gate
    assert fit.assess(dream, hardware=hw(ram_total_gb=8.0)).blocking_reason() is None


# -- the refusal chain ---------------------------------------------------------


class FakeAdapter:
    def __init__(self, recipe):
        self.recipe = recipe
        self.loaded = True

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False


@pytest.fixture()
def tiny_gpu(monkeypatch):
    monkeypatch.setattr(
        fit, "detect_hardware", lambda refresh=False: hw(vram_total_gb=8.0, vram_free_gb=7.5)
    )


def test_manager_refuses_before_creating_adapter(monkeypatch, tiny_gpu):
    import ollada.runtime.manager as manager_mod

    created = []
    monkeypatch.setattr(manager_mod, "create_adapter", lambda r: created.append(r) or FakeAdapter(r))
    manager = manager_mod.ModelManager()
    with pytest.raises(fit.FitError):
        manager.ensure_loaded(get_recipe("dream:7b"))  # 9.80 GB peak, 8 GB card
    assert created == []  # refused before any adapter existed
    adapter = manager.ensure_loaded(get_recipe("dream:7b"), force=True)
    assert adapter.loaded is True


def test_api_maps_fit_refusal_to_409_and_force_overrides(monkeypatch, tiny_gpu):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ollada.runtime.manager as manager_mod
    from ollada.adapters.base import GenResult
    from ollada.server import create_app

    class Chatty(FakeAdapter):
        def chat(self, messages, overrides=None):
            return GenResult(text="ok", tokens=1, seconds=0.01)

    monkeypatch.setattr(manager_mod, "create_adapter", lambda r: Chatty(r))
    client = TestClient(create_app(manager_mod.ModelManager()), raise_server_exceptions=False)
    refused = client.post(
        "/api/chat", json={"model": "dream:7b", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert refused.status_code == 409
    forced = client.post(
        "/api/chat",
        json={
            "model": "dream:7b",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"force": True},
        },
    )
    assert forced.status_code == 200
