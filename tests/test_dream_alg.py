"""Dream alg passthrough tests.

Dream's upstream diffusion_generate already ships confidence-ranked
token transfer (alg="maskgit_plus" / "topk_margin" / "entropy") next
to the random-transfer default ("origin") — the confidence-aware
parallel decoding half of Fast-dLLM (arXiv:2505.22618), in the
weights' own code. ollada's adapter used to hardcode alg="origin";
it now reads alg/alg_temp from recipe params, defaulting to exactly
today's behavior.

Torch-free by design (same stub pattern as test_sdar_cpu.py): the
model and tokenizer are fakes, and _prompt_ids/_finish are stubbed
on the instance so the test pins only the kwargs chat() passes to
diffusion_generate.
"""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest


@contextlib.contextmanager
def _stubbed_dream():
    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "fp16"
    torch_stub.bfloat16 = "bf16"
    torch_stub.no_grad = lambda: (lambda f: f)
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModel = object
    transformers_stub.AutoTokenizer = object
    stubs = {"torch": torch_stub, "transformers": transformers_stub}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("ollada.adapters.dream", None)
    try:
        import ollada.adapters.dream as dream

        yield dream
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sys.modules.pop("ollada.adapters.dream", None)


class _FakeIds:
    shape = (1, 4)

    def to(self, _device):
        return self


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.config = SimpleNamespace(mask_token_id=5)
        self.calls = []

    def diffusion_generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=[])


def _adapter(dream, generate):
    from ollada.adapters.dream import DreamAdapter

    recipe = SimpleNamespace(name="dream:7b", generate=dict(generate))
    adapter = DreamAdapter(recipe)
    adapter.model = _FakeModel()
    adapter.tokenizer = SimpleNamespace()
    adapter._prompt_ids = lambda _messages: _FakeIds()
    adapter._finish = lambda _ids, _plen, _started: SimpleNamespace(text="ok")
    return adapter


def test_default_alg_is_origin():
    with _stubbed_dream() as dream:
        adapter = _adapter(dream, {"gen_length": 256, "steps": 128})
        adapter.chat([{"role": "user", "content": "hi"}])
        assert adapter.model.calls[0]["alg"] == "origin"
        assert adapter.model.calls[0]["alg_temp"] == 0.0
        assert adapter.model.calls[0]["steps"] == 128


def test_maskgit_plus_passes_through():
    with _stubbed_dream() as dream:
        adapter = _adapter(dream, {"gen_length": 256, "steps": 64})
        adapter.chat(
            [{"role": "user", "content": "hi"}],
            overrides={"alg": "maskgit_plus", "steps": 64},
        )
        assert adapter.model.calls[0]["alg"] == "maskgit_plus"
        assert adapter.model.calls[0]["steps"] == 64


def test_unknown_alg_fails_before_the_model_runs():
    with _stubbed_dream() as dream:
        from ollada.adapters.base import AdapterError

        adapter = _adapter(dream, {"gen_length": 256, "steps": 128})
        with pytest.raises(AdapterError, match="unknown Dream alg"):
            adapter.chat(
                [{"role": "user", "content": "hi"}],
                overrides={"alg": "nope"},
            )
        assert adapter.model.calls == []
