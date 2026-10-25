"""ModelManager watch-gate tests.

The step callback (watch mode) must reach only adapters that
support it. Forwarding it blindly would change the call shape for
the other families' adapters, whose chat() predates watch mode.
"""

from types import SimpleNamespace

from ollada.adapters.base import GenResult
from ollada.runtime.manager import ModelManager


class _PlainAdapter:
    supports_watch = False

    def __init__(self):
        self.recipe = SimpleNamespace(name="fake:1")
        self.calls = []

    def chat(self, messages, overrides=None):
        self.calls.append((messages, overrides))
        return GenResult(text="ok", tokens=0, seconds=0.0)


class _WatchAdapter:
    supports_watch = True

    def __init__(self):
        self.recipe = SimpleNamespace(name="fake:2")
        self.calls = []

    def chat(self, messages, overrides=None, on_step=None):
        self.calls.append((messages, overrides, on_step))
        return GenResult(text="ok", tokens=0, seconds=0.0)


def _manager_with(adapter):
    manager = ModelManager()
    manager.ensure_loaded = lambda name, force=False: adapter
    return manager


def test_step_callback_not_forwarded_to_plain_adapters():
    adapter = _PlainAdapter()
    manager = _manager_with(adapter)
    result = manager.chat("fake:1", [{"role": "user", "content": "hi"}], on_step=object())
    assert result.text == "ok"
    # Called with the pre-watch shape exactly: two positional args.
    assert adapter.calls == [([{"role": "user", "content": "hi"}], None)]


def test_step_callback_forwarded_to_watch_adapters():
    adapter = _WatchAdapter()
    manager = _manager_with(adapter)
    marker = object()
    result = manager.chat("fake:2", [{"role": "user", "content": "hi"}], on_step=marker)
    assert result.text == "ok"
    assert adapter.calls[0][2] is marker


def test_no_callback_keeps_the_plain_call():
    adapter = _WatchAdapter()
    manager = _manager_with(adapter)
    manager.chat("fake:2", [{"role": "user", "content": "hi"}])
    assert adapter.calls[0][2] is None
