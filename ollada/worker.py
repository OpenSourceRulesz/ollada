"""Worker subprocess: one model family, one pinned environment.

The manager spawns `python -m ollada.worker` from a family's worker venv
(see ollada/workers.py) and talks to it over stdin/stdout with one JSON
object per line. Stdout is protocol-only: adapter code runs with stdout
redirected to stderr, because at least one upstream modeling file prints
to stdout during load (SDAR's auto_docstring scolding) and a stray line
would corrupt the stream. Logs, warnings, and progress bars all go to
stderr by convention — nothing legitimate needs stdout here.

Protocol (manager -> worker, one request per line):
  {"id": 1, "op": "load", "model": "dream:7b"}
  {"id": 2, "op": "chat", "messages": [...], "overrides": {...}|null}
  {"id": 3, "op": "status"}

Responses (worker -> manager):
  {"id": 1, "ok": true, "result": {"loaded": ..., "family": ...,
      "supports_watch": bool, "peak_gb": float|null, "hardware": str|null}}
  {"id": 2, "ok": true, "result": {"text": ..., "tokens": ..., "seconds": ...,
      "nfe": ...|null, "peak_gb": ..., "hardware": ...}}
  {"id": 3, "ok": true, "result": {"loaded": ...|null, "family": ...|null}}
  {"id": 2, "ok": false, "error": "..."}
  {"event": "step", "pieces": ..., "block_i": ..., "num_blocks": ...,
      "step_i": ..., "steps": ...}   # only between a chat request and its result

The worker prints {"event": "ready", "pid": ...} on startup so the manager
can tell a broken spawn (bad venv, missing package) from a slow load.
EOF on stdin ends the loop and the process exits; the manager also kills
the process on unload, which is the clean-VRAM-unload half of the design.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys


class WorkerError(RuntimeError):
    """Raised when the worker protocol itself breaks (not a model failure)."""


def _cuda_peak() -> tuple[float | None, str | None]:
    """(peak GB since last reset, device name), or (None, None) without CUDA.

    Everything is guarded: the worker must report "no peak" on a CPU box,
    never crash the protocol because torch is absent or CUDA is not.
    """
    try:
        import torch
    except Exception:
        return None, None
    try:
        if not torch.cuda.is_available():
            return None, None
        peak = float(torch.cuda.max_memory_allocated()) / (1024**3)
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return peak, str(torch.cuda.get_device_name(0))
    except Exception:
        return None, None


def main(argv=None, adapter_factory=None, stdin=None, stdout=None) -> int:
    """Run the request loop. Returns 0 on stdin EOF.

    adapter_factory is a test seam: production passes None and adapters
    come from the registry; tests inject a stub that needs no torch.
    stdin/stdout default to the process streams; tests pass StringIO.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    def emit(obj: dict) -> None:
        stdout.write(json.dumps(obj) + "\n")
        stdout.flush()

    emit({"event": "ready", "pid": os.getpid()})

    # Imported lazily so a broken worker env fails at first use with a
    # clear error response, not at process start with a bare traceback.
    from ollada.adapters import create_adapter
    from ollada.recipes import get_recipe

    adapter = None
    loaded_name: str | None = None

    def step_emit(pieces, block_i, num_blocks, step_i, steps) -> None:
        emit({
            "event": "step",
            "pieces": pieces,
            "block_i": block_i,
            "num_blocks": num_blocks,
            "step_i": step_i,
            "steps": steps,
        })

    for line in stdin:
        try:
            req = json.loads(line)
        except ValueError:
            emit({"id": None, "ok": False, "error": f"unparseable request line: {line[:80]!r}"})
            continue
        if not isinstance(req, dict) or "op" not in req:
            emit({"id": req.get("id") if isinstance(req, dict) else None,
                  "ok": False, "error": "request must be an object with an 'op'"})
            continue
        req_id = req.get("id")
        op = req["op"]
        try:
            if op == "load":
                recipe = get_recipe(req["model"])
                adapter = adapter_factory(recipe) if adapter_factory else create_adapter(recipe)
                # Stdout is protocol-only: see the module docstring.
                with contextlib.redirect_stdout(sys.stderr):
                    adapter.load()
                loaded_name = recipe.name
                peak, hardware = _cuda_peak()
                emit({"id": req_id, "ok": True, "result": {
                    "loaded": loaded_name,
                    "family": recipe.family,
                    "supports_watch": bool(getattr(adapter, "supports_watch", False)),
                    "peak_gb": peak,
                    "hardware": hardware,
                }})
            elif op == "chat":
                if adapter is None:
                    raise WorkerError("chat before load")
                kwargs: dict = {}
                if getattr(adapter, "supports_watch", False):
                    kwargs["on_step"] = step_emit
                with contextlib.redirect_stdout(sys.stderr):
                    result = adapter.chat(req["messages"], req.get("overrides"), **kwargs)
                peak, hardware = _cuda_peak()
                emit({"id": req_id, "ok": True, "result": {
                    "text": result.text,
                    "tokens": result.tokens,
                    "seconds": result.seconds,
                    "nfe": result.nfe,
                    "peak_gb": peak,
                    "hardware": hardware,
                }})
            elif op == "status":
                emit({"id": req_id, "ok": True, "result": {
                    "loaded": loaded_name,
                    "family": adapter.recipe.family if adapter is not None else None,
                }})
            else:
                emit({"id": req_id, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001 — the protocol must report, not die
            emit({"id": req_id, "ok": False,
                  "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
