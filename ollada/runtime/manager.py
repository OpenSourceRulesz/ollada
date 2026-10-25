"""Model manager — one model loaded at a time, serial generation.

Since 0.3.0 the manager no longer loads adapters in-process: each family
runs in an isolated worker subprocess with its own pinned environment
(ollada/worker.py speaks the protocol, ollada/workers.py owns the venvs).
The manager keeps the interface the CLI and server already use, so neither
changes: ensure_loaded / chat / unload / status, one GPU, one generation
at a time, fit refusal before anything is touched.

What the worker design buys over the 0.2.0 in-process manager:
- incompatible transformers pins stop colliding (dream 4.x, diffcoder
  5.3.0, sdar >=4.57 each get their own env — no more pin dance);
- crash isolation (a model segfault kills the worker, not the daemon);
- clean VRAM unload (killing the worker frees the card for certain).

Escape hatch: OLLADA_INPROCESS=1 restores the old in-process path (debug
and tests only — it reintroduces the pin collision, so it is never the
default). The 0.2.0 fail-fast pin check still guards that path.

Worker lifecycle mirrors the old manager lifecycle: a worker is spawned
on load and killed on unload or model switch. If the manager process
dies, the worker exits on stdin EOF; an atexit hook kills stragglers.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from ollada import fit, measurements, workers
from ollada.adapters import create_adapter
from ollada.adapters.base import AdapterError, BaseAdapter, GenResult
from ollada.recipes import Recipe, env_mismatch, get_recipe
from ollada.worker import WorkerError

_READY_TIMEOUT_S = 120  # worker startup is imports only; a load takes minutes, startup must not
_KILL_TIMEOUT_S = 10

_LIVE: set[subprocess.Popen] = set()


def _kill_all_live() -> None:
    for proc in list(_LIVE):
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in list(_LIVE):
        try:
            proc.wait(timeout=_KILL_TIMEOUT_S)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _LIVE.clear()


atexit.register(_kill_all_live)


class WorkerAdapter(BaseAdapter):
    """Manager-side handle for one worker subprocess.

    It implements the BaseAdapter interface over the JSON-lines protocol,
    so the manager (and its tests) treat a worker like any adapter. The
    model itself lives in the worker; self.model stays None and `loaded`
    tracks the worker's state instead.
    """

    def __init__(self, recipe: Recipe, worker_python: Path):
        super().__init__(recipe)
        self._worker_python = str(worker_python)
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._loaded = False
        self._next_id = 0
        self._step_handler = None
        self.supports_watch = False  # set from the worker's load result

    # -- process plumbing -------------------------------------------------

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        except Exception:
            pass
        finally:
            self._lines.put(None)  # EOF sentinel: the worker is gone

    def _start(self) -> None:
        env = dict(os.environ)
        ollada_root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = ollada_root + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            [self._worker_python, "-m", "ollada.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # worker logs go to the manager's stderr: visible, not protocol
            text=True,
            bufsize=1,
            env=env,
        )
        _LIVE.add(self._proc)
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        try:
            first = self._lines.get(timeout=_READY_TIMEOUT_S)
        except queue.Empty:
            raise WorkerError("worker did not print its ready line "
                              f"(is {self._worker_python} a working venv python?)") from None
        if first is None:
            raise WorkerError(f"worker exited during startup (rc={self._proc.poll()})")
        try:
            ready = json.loads(first)
        except ValueError:
            raise WorkerError(f"worker printed a non-JSON first line: {first[:80]!r}") from None
        if ready.get("event") != "ready":
            raise WorkerError(f"worker first line was not a ready event: {ready!r}")

    def _send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError(f"worker stdin closed (rc={self._proc.poll()})") from exc

    def _call(self, op: str, **fields) -> dict:
        """Send one request; return its result, forwarding step events.

        Raises AdapterError on a worker error response, WorkerError if the
        worker dies mid-request. Only one request is outstanding at a
        time (the manager holds its lock), so no id correlation is needed
        beyond matching the response id.
        """
        self._next_id += 1
        req_id = self._next_id
        self._send({"id": req_id, **fields, "op": op})
        while True:
            line = self._lines.get()
            if line is None:
                rc = self._proc.poll() if self._proc else None
                raise WorkerError(f"worker exited mid-request (rc={rc})")
            try:
                msg = json.loads(line)
            except ValueError:
                raise WorkerError(f"worker printed a non-JSON line: {line[:80]!r}") from None
            if not isinstance(msg, dict):
                raise WorkerError(f"worker printed a non-object line: {line[:80]!r}")
            if msg.get("event") == "step" and self._step_handler is not None:
                self._step_handler(
                    msg.get("pieces"), msg.get("block_i"), msg.get("num_blocks"),
                    msg.get("step_i"), msg.get("steps"))
                continue
            if msg.get("id") != req_id:
                continue  # stale or unsolicited; the protocol has no other traffic
            if not msg.get("ok", False):
                raise AdapterError(str(msg.get("error", "worker error")))
            return msg["result"]

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        _LIVE.discard(proc)
        try:
            proc.terminate()
            proc.wait(timeout=_KILL_TIMEOUT_S)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

    # -- BaseAdapter interface --------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.poll() is None

    def load(self) -> dict:
        """Spawn the worker and load the model in it. Returns the load result."""
        self._start()
        try:
            result = self._call("load", model=self.recipe.name)
        except Exception:
            self._kill()
            raise
        self._loaded = True
        self.supports_watch = bool(result.get("supports_watch", False))
        return result

    def unload(self) -> None:
        self._kill()
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def chat(
        self,
        messages: list[dict],
        overrides: dict | None = None,
        on_step=None,
    ) -> GenResult:
        if not self.loaded:
            raise AdapterError("chat before load")
        self._step_handler = on_step
        try:
            result = self._call("chat", messages=messages, overrides=overrides)
        finally:
            self._step_handler = None
        return GenResult(
            text=result["text"],
            tokens=result["tokens"],
            seconds=result["seconds"],
            nfe=result.get("nfe"),
            peak_gb=result.get("peak_gb"),
            hardware=result.get("hardware"),
        )


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._adapter: BaseAdapter | None = None

    @property
    def current_name(self) -> str | None:
        return self._adapter.recipe.name if self._adapter else None

    def status(self) -> dict:
        return {
            "loaded": self._adapter.recipe.name if self._adapter else None,
            "family": self._adapter.recipe.family if self._adapter else None,
        }

    def ensure_loaded(self, recipe_or_name, force: bool = False) -> BaseAdapter:
        recipe: Recipe = (
            recipe_or_name if isinstance(recipe_or_name, Recipe) else get_recipe(recipe_or_name)
        )
        if self._adapter and self._adapter.recipe.name == recipe.name and self._adapter.loaded:
            return self._adapter
        if not force:
            # Refuse BEFORE unloading the resident model or touching the GPU:
            # a refused load leaves the current state exactly as it was.
            # The fit check runs before any worker env is built, so a
            # refused model never pays the one-time env download either.
            reason = fit.assess(recipe).blocking_reason()
            if reason:
                raise fit.FitError(reason)
        if os.environ.get("OLLADA_INPROCESS"):
            if not force:
                # The 0.2.0 fail-fast pin check, kept for the debug path.
                mismatch = env_mismatch(recipe.env)
                if mismatch:
                    raise fit.FitError(mismatch)
            if self._adapter:
                self._adapter.unload()
                self._adapter = None
            adapter = create_adapter(recipe)
            measurements.reset_cuda_peak()
            adapter.load()
            self._adapter = adapter
            measurements.capture_and_record(recipe.name)
            return adapter
        # Env build happens before evicting the resident model: a build
        # failure must not cost the currently loaded model.
        worker_python = workers.ensure_env(recipe)
        if self._adapter:
            self._adapter.unload()
            self._adapter = None
        adapter = WorkerAdapter(recipe, worker_python)
        load_result = adapter.load()
        self._adapter = adapter
        if load_result.get("peak_gb") is not None:
            measurements.record_peak(
                recipe.name, load_result["peak_gb"], load_result.get("hardware"))
        return adapter

    def chat(
        self,
        name: str,
        messages: list[dict],
        overrides: dict | None = None,
        force: bool = False,
        on_step=None,
    ) -> GenResult:
        with self._lock:  # one GPU, one generation at a time
            adapter = self.ensure_loaded(name, force=force)
            # Watch mode: the step callback reaches only adapters that
            # support it — the same gate as before the worker rewrite
            # (the worker enforces it again on its side, belt and braces).
            if on_step is not None and getattr(adapter, "supports_watch", False):
                result = adapter.chat(messages, overrides, on_step=on_step)
            else:
                result = adapter.chat(messages, overrides)
            if result.peak_gb is not None:
                measurements.record_peak(name, result.peak_gb, result.hardware)
            else:
                # In-process path: peak comes from the local torch.
                measurements.capture_and_record(name)
            return result

    def unload(self) -> None:
        with self._lock:
            if self._adapter:
                self._adapter.unload()
                self._adapter = None
