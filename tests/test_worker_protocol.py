"""Worker protocol tests — no torch, no GPU, no network.

The worker loop is driven over StringIO with stub adapters, so these
prove the protocol contract itself: load/chat/status shapes, step-event
forwarding, error reporting, and the stdout guard (a stub that prints
to stdout mid-load must not corrupt the stream — the SDAR upstream
print is the real-world case).
"""

import io
import json

import pytest

from ollada.adapters.base import GenResult
from ollada.recipes import get_recipe
from ollada.worker import main as worker_main


class _StubAdapter:
    supports_watch = True

    def __init__(self, recipe):
        self.recipe = recipe
        self.loaded = False
        self.calls = []

    def load(self):
        print("stray stdout during load (must not corrupt the protocol)")
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat(self, messages, overrides=None, on_step=None):
        print("stray stdout during chat (must not corrupt the protocol)")
        if on_step is not None:
            on_step(["hel"], 0, 1, 0, 2)
            on_step(["hello"], 0, 1, 1, 2)
        self.calls.append((messages, overrides))
        return GenResult(text="echo: " + messages[-1]["content"], tokens=3, seconds=0.5, nfe=9)


class _PlainAdapter(_StubAdapter):
    supports_watch = False

    def chat(self, messages, overrides=None, on_step=None):
        assert on_step is None  # worker must not pass on_step to plain adapters
        return GenResult(text="plain", tokens=1, seconds=0.1)


class _ExplodingAdapter(_StubAdapter):
    def load(self):
        raise RuntimeError("boom on load")


def _run_worker(requests, factory):
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    rc = worker_main(adapter_factory=factory, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return rc, lines


def test_ready_first_then_load_chat_status():
    rc, lines = _run_worker([
        {"id": 1, "op": "load", "model": "dream:7b"},
        {"id": 2, "op": "chat", "messages": [{"role": "user", "content": "hi"}], "overrides": None},
        {"id": 3, "op": "status"},
    ], _StubAdapter)
    assert rc == 0
    assert lines[0]["event"] == "ready" and "pid" in lines[0]
    load = lines[1]
    assert load == {"id": 1, "ok": True, "result": {
        "loaded": "dream:7b", "family": "dream", "supports_watch": True,
        "peak_gb": None, "hardware": None}}
    steps = [m for m in lines if m.get("event") == "step"]
    assert len(steps) == 2
    assert steps[0]["pieces"] == ["hel"] and steps[1]["step_i"] == 1
    chat = [m for m in lines if m.get("id") == 2][0]
    assert chat["ok"] is True
    assert chat["result"]["text"] == "echo: hi"
    assert chat["result"]["nfe"] == 9
    status = [m for m in lines if m.get("id") == 3][0]
    assert status["result"] == {"loaded": "dream:7b", "family": "dream"}


def test_stdout_prints_never_reach_the_protocol():
    # The stub prints on load and chat; every stdout line must still
    # parse as JSON, or the manager would choke on a stray print.
    _, lines = _run_worker([
        {"id": 1, "op": "load", "model": "dream:7b"},
        {"id": 2, "op": "chat", "messages": [{"role": "user", "content": "hi"}], "overrides": None},
    ], _StubAdapter)
    assert all(isinstance(m, dict) for m in lines)


def test_plain_adapter_gets_no_on_step_and_no_step_events():
    _, lines = _run_worker([
        {"id": 1, "op": "load", "model": "dream:7b"},
        {"id": 2, "op": "chat", "messages": [{"role": "user", "content": "hi"}], "overrides": None},
    ], _PlainAdapter)
    assert not [m for m in lines if m.get("event") == "step"]
    load = [m for m in lines if m.get("id") == 1][0]
    assert load["result"]["supports_watch"] is False


def test_adapter_failure_is_an_error_response_not_a_crash():
    _, lines = _run_worker([
        {"id": 1, "op": "load", "model": "dream:7b"},
        {"id": 2, "op": "status"},
    ], _ExplodingAdapter)
    load = [m for m in lines if m.get("id") == 1][0]
    assert load["ok"] is False and "boom on load" in load["error"]
    # The loop survives the failure and answers the next request.
    status = [m for m in lines if m.get("id") == 2][0]
    assert status["ok"] is True


def test_unknown_op_and_chat_before_load_are_errors():
    _, lines = _run_worker([
        {"id": 1, "op": "teleport"},
        {"id": 2, "op": "chat", "messages": [], "overrides": None},
    ], _StubAdapter)
    assert [m for m in lines if m.get("id") == 1][0]["ok"] is False
    assert [m for m in lines if m.get("id") == 2][0]["ok"] is False


def test_real_recipe_loads_through_the_factory():
    # get_recipe runs inside the worker; the factory only swaps the adapter.
    _, lines = _run_worker([{"id": 1, "op": "load", "model": "sdar:4b"}], _StubAdapter)
    load = [m for m in lines if m.get("id") == 1][0]
    assert load["result"]["family"] == "sdar"
    assert get_recipe("sdar:4b").family == "sdar"
