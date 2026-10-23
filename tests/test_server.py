"""API shape tests with a fake adapter — unit level only.

These prove the HTTP contract, NOT that any model runs. Model support is
proven by the real-load verification gate (PLAN.md section 7).
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ollada.adapters.base import GenResult  # noqa: E402
from ollada.runtime.manager import ModelManager  # noqa: E402
from ollada.server import create_app  # noqa: E402


class FakeAdapter:
    def __init__(self, recipe):
        self.recipe = recipe
        self.loaded = True

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat(self, messages, overrides=None):
        return GenResult(text=f"echo: {messages[-1]['content']}", tokens=4, seconds=0.01, nfe=7)


@pytest.fixture()
def client(monkeypatch):
    import ollada.runtime.manager as manager_mod
    from ollada import fit

    monkeypatch.setattr(manager_mod, "create_adapter", lambda recipe: FakeAdapter(recipe))
    # Shape tests run on whatever machine pytest runs on; pin fit detection
    # to a capable machine so they test HTTP shape, not the fit gate (the
    # gate has its own tests in test_fit.py, including the 409 mapping).
    monkeypatch.setattr(
        fit,
        "detect_hardware",
        lambda refresh=False: fit.Hardware(
            gpu_name="Test GPU", vram_total_gb=24.0, vram_free_gb=23.0, ram_total_gb=64.0
        ),
    )
    return TestClient(create_app(ModelManager()))


def test_health_and_tags(client):
    assert client.get("/health").json()["status"] == "ok"
    names = {m["name"] for m in client.get("/api/tags").json()["models"]}
    assert names == {"dream:7b", "sdar:4b", "diffcoder:8b", "sdar:1.7b"}


def test_ollama_shaped_chat(client):
    resp = client.post(
        "/api/chat",
        json={"model": "dream:7b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["content"] == "echo: hi"
    assert body["nfe"] == 7


def test_openai_shaped_chat(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "sdar:4b", "messages": [{"role": "user", "content": "yo"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "echo: yo"


def test_unknown_model_is_404(client):
    resp = client.post(
        "/api/chat",
        json={"model": "nope:1b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
