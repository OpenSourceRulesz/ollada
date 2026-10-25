"""Worker environments: one venv per model family, pinned from its recipe.

Why a venv per family instead of one shared env: the families' pins are
incompatible (dream needs transformers 4.x, diffcoder needs 5.3.0, sdar
needs >=4.57), so no single install satisfies the shelf. A worker venv
holds exactly one family's stack; the manager spawns the venv's python
with PYTHONPATH pointed at the manager's ollada package, so the worker
code itself always matches the manager's version and is never installed
twice.

Layout: ~/.ollada/workers/<family>/ (OLLADA_WORKERS_DIR overrides, same
pattern as OLLADA_MEASUREMENTS_DIR). Envs are built lazily on first load
of a family — the first load downloads torch, so it is slow once and
says so. A pin mismatch in an existing env is a refusal, not a silent
rebuild: rebuilding re-downloads gigabytes, so it happens only on an
explicit `ollada workers rebuild <model>`.

Torch index: on a machine with no NVIDIA GPU (nvidia-smi absent), torch
comes from the CPU index instead of the ~2GB CUDA bundle — the same tip
the README gives for manual installs, applied automatically. A GPU
machine uses the default index.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ollada import fit
from ollada.recipes import Recipe, spec_satisfied

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


class WorkerEnvError(RuntimeError):
    """Raised when a worker env cannot be built, checked, or found."""


def workers_dir() -> Path:
    override = os.environ.get("OLLADA_WORKERS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".ollada" / "workers"


def venv_path(family: str) -> Path:
    return workers_dir() / family


def venv_python(family: str) -> Path:
    root = venv_path(family)
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def installed_transformers(python: Path) -> str | None:
    """Installed transformers version in a venv, or None if absent/broken.

    Uses importlib.metadata, not an import: importing transformers is
    slow and can fail for reasons unrelated to the version question.
    """
    if not Path(python).exists():
        return None
    try:
        proc = subprocess.run(
            [str(python), "-c",
             "import importlib.metadata as m; print(m.version('transformers'))"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    version = proc.stdout.strip()
    return version or None


def _pip_packages(recipe: Recipe) -> list[str]:
    """Install set for a family's worker env, order-deduped.

    pyyaml (the worker imports ollada.recipes), torch, the recipe's
    transformers pin, accelerate, then the recipe's extras. Extras may
    repeat torch/accelerate — dedupe keeps the first occurrence.
    """
    spec = (recipe.env or {}).get("transformers", "")
    pkgs = ["pyyaml", "torch", f"transformers{spec}", "accelerate"]
    pkgs.extend((recipe.env or {}).get("extras", []) or [])
    seen: set[str] = set()
    out: list[str] = []
    for pkg in pkgs:
        key = pkg.split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip()
        if key not in seen:
            seen.add(key)
            out.append(pkg)
    return out


def _run_checked(argv: list[str], what: str, _run=subprocess.run) -> None:
    try:
        proc = _run(argv, capture_output=True, text=True)
    except Exception as exc:
        raise WorkerEnvError(f"could not {what}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise WorkerEnvError(
            f"could not {what} (exit {proc.returncode}): " + " | ".join(tail))


def ensure_env(recipe: Recipe, _run=subprocess.run) -> Path:
    """Return the venv python for a family's worker env, building if needed.

    Raises WorkerEnvError on a build failure or a pin mismatch in an
    existing env (rebuild explicitly; see rebuild_env). _run is a test
    seam for the subprocess calls.
    """
    family = recipe.family
    python = venv_python(family)
    spec = str((recipe.env or {}).get("transformers", ""))
    if python.exists():
        if spec:
            installed = installed_transformers(python)
            if installed is not None and not spec_satisfied(spec, installed):
                raise WorkerEnvError(
                    f"worker env for {family} has transformers {installed}, "
                    f"but {recipe.name} pins transformers{spec}. Rebuild it: "
                    f"ollada workers rebuild {recipe.name}")
        return python
    # Fresh build. _run is a test seam for every subprocess call here so
    # tests can observe the venv creation and pip installs without network.
    print(f"building worker env for {family} (one-time download) ...")
    try:
        proc = _run([sys.executable, "-m", "venv", str(venv_path(family))],
                    capture_output=True, text=True)
    except Exception as exc:
        raise WorkerEnvError(f"could not create worker env for {family}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise WorkerEnvError(
            f"could not create worker env for {family}: " + " | ".join(tail))
    pkgs = _pip_packages(recipe)
    hardware = fit.detect_hardware()
    if hardware.gpu_name is None:
        # No NVIDIA GPU: torch from the CPU index, everything else from
        # PyPI (a single --index-url would move all packages, not just torch).
        torch_pkgs = [p for p in pkgs if p == "torch"]
        rest = [p for p in pkgs if p != "torch"]
        _run_checked([str(python), "-m", "pip", "install",
                      "--index-url", CPU_TORCH_INDEX, *torch_pkgs],
                     f"install torch (CPU index) for {family}", _run)
        if rest:
            _run_checked([str(python), "-m", "pip", "install", *rest],
                         f"install worker packages for {family}", _run)
    else:
        _run_checked([str(python), "-m", "pip", "install", *pkgs],
                     f"install worker packages for {family}", _run)
    return python


def rebuild_env(recipe: Recipe, _run=subprocess.run) -> Path:
    """Delete a family's worker env and rebuild it from the recipe."""
    shutil.rmtree(venv_path(recipe.family), ignore_errors=True)
    return ensure_env(recipe, _run)


def list_envs(recipes: list[Recipe]) -> list[dict]:
    """One row per family: path, built?, installed transformers version."""
    rows = []
    for family in sorted({r.family for r in recipes}):
        python = venv_python(family)
        rows.append({
            "family": family,
            "path": str(venv_path(family)),
            "built": python.exists(),
            "transformers": installed_transformers(python) if python.exists() else None,
        })
    return rows
