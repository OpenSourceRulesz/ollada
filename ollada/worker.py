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

Worker isolation note: the manager spawns this process with PYTHONPATH
pointing at the manager's ollada location, so the worker runs the
manager's exact code. Those entries are stripped from sys.path on
startup (see _drop_manager_paths_from_sys_path) once the ollada
package itself is imported, so the worker's third-party packages always
resolve from its own pinned venv, never from the manager's venv.
"""

from __future__ import annotations

import contextlib
import json
import os
import site
import sys
import traceback
from pathlib import Path


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


def _drop_manager_paths_from_sys_path() -> list[str]:
    """Drop the manager's PYTHONPATH entries (and cwd) from sys.path.

    The manager spawns the worker with PYTHONPATH pointed at the
    manager's ollada location so the worker runs the manager's exact
    code. For an installed manager that location is site-packages —
    which would also put the manager venv's packages (torch,
    transformers, ...) ahead of the worker venv's pinned ones on
    sys.path, silently breaking worker isolation. Proven 2026-10-27:
    a PyPI-installed manager's unpinned transformers 5.21.0 shadowed
    the sdar worker env's pinned 4.57.6, and under 5.x SDAR's
    rope_scaling came out None, so its modeling fell back to a
    "default" rope type that does not exist (KeyError). Repo and
    editable managers were immune — their ollada location is the repo
    root, which holds no third-party packages.

    This runs only in the real worker subprocess (the __main__ block),
    never when main() is driven in-process by tests: by then the
    ollada package itself is already imported (python -m imports it
    before this runs), and submodule imports resolve through the
    package's __path__, not sys.path — so dropping these entries
    cannot break ollada imports. It only changes where third-party
    packages (transformers, torch) resolve from: the worker venv.
    The worker venv's own site-packages are exempted, so a degenerate
    setup where manager and worker share one venv keeps working (and
    there is nothing to shadow in that case — same env, same versions).

    Returns the removed entries, for tests and debugging.
    """
    removed: list[str] = []
    # The worker venv's own site-packages must never be stripped.
    exempt: set[str] = set()
    try:
        for entry in site.getsitepackages():
            exempt.add(str(Path(entry).resolve()))
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            exempt.add(str(Path(user_site).resolve()))
    except Exception:
        pass

    candidates: set[str] = set()
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            candidates.add(str(Path(entry).resolve()))
        except Exception:
            candidates.add(entry)

    cwd = str(Path.cwd().resolve())
    kept: list[str] = []
    for entry in sys.path:
        # '' means cwd, and for `python -m` sys.path[0] is cwd: the
        # worker never imports from the caller's working directory.
        if entry == "" or entry == cwd:
            removed.append(entry)
            continue
        try:
            resolved = str(Path(entry).resolve())
        except Exception:
            resolved = entry
        if resolved in candidates and resolved not in exempt:
            removed.append(entry)
            continue
        kept.append(entry)
    sys.path[:] = kept
    # Do not leak the manager's path to anything the worker spawns.
    os.environ.pop("PYTHONPATH", None)
    return removed


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
            # Include the traceback: the manager-side error used to carry
            # only the message, which made worker failures (like the Dell
            # SDARConfig crash) undebuggable from the CLI. The traceback
            # goes to the error output, never to stdout's protocol lines
            # beyond this JSON object.
            emit({"id": req_id, "ok": False,
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    # Isolate BEFORE any adapter/third-party import: main() imports
    # adapters lazily, so this ordering is what makes the strip safe.
    _drop_manager_paths_from_sys_path()
    raise SystemExit(main())
