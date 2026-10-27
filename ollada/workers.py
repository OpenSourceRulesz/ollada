"""Worker environments: one venv per model family, pinned from its recipe.

Why a venv per family instead of one shared env: the families' pins are
incompatible (dream needs transformers 4.x, diffcoder needs 5.3.0, sdar
needs >=4.57,<5), so no single install satisfies the shelf. A worker venv
holds exactly one family's stack; the manager spawns the venv's python
with PYTHONPATH pointed at the manager's ollada package, so the worker
code itself always matches the manager's version and is never installed
twice. The worker drops those PYTHONPATH entries from sys.path once
its own ollada package is imported (ollada/worker.py), so manager-env
packages cannot shadow the worker env's pins — an installed manager's
unpinned transformers once did exactly that (2026-10-27).

One env per family does NOT mean one model's recipe: the install set is
the UNION of the family's recipes' extras. The sdar family proved why —
1.7b's modeling never imports einops, 4b's does, and an env built from
the 1.7b recipe alone (the first run) left the 4b unloadable with a bare
ImportError (found on the Dell, 2026-10-25). The transformers pin stays
per requested model, and a fresh build verifies the installed version
satisfies EVERY family recipe's pin, so incompatible family pins fail
loudly at build time instead of thrashing between rebuilds.

Layout: ~/.ollada/workers/<family>/ (OLLADA_WORKERS_DIR overrides, same
pattern as OLLADA_MEASUREMENTS_DIR). Envs are built lazily on first load
of a family — the first load downloads torch, so it is slow once and
says so. A pin mismatch in an existing env is a refusal, not a silent
rebuild: rebuilding re-downloads gigabytes, so it happens only on an
explicit `ollada workers rebuild <model>`. A missing package in an
existing env (built before the union rule, or by hand) refuses the same
way, with the same rebuild pointer.

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
from ollada.recipes import Recipe, load_catalog, spec_satisfied

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# PyPI's torch is CPU-only (2.14.1+cpu, no nvidia packages) — a GPU
# machine that installs torch from PyPI silently lands CPU torch, as
# the desktop dream env did. CUDA wheels live on PyTorch's index.
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


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


def _norm_dist_name(name: str) -> str:
    return name.lower().replace("-", "_")


def installed_distributions(python: Path) -> set[str] | None:
    """Installed distribution names in a venv, normalized, or None.

    One importlib.metadata scan, no imports of the packages themselves
    (importing torch just to ask if it is installed is slow and can fail
    for unrelated reasons). None means the venv python is missing/broken.
    """
    if not Path(python).exists():
        return None
    try:
        proc = subprocess.run(
            [str(python), "-c",
             "import importlib.metadata as m; "
             "print('\\n'.join(d.metadata['Name'] for d in m.distributions()))"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return {_norm_dist_name(line) for line in proc.stdout.splitlines() if line.strip()}


def _family_recipes(family: str) -> list[Recipe]:
    try:
        return [r for r in load_catalog() if r.family == family]
    except Exception:
        return []


def _family_extras(family: str) -> list[str]:
    """Union of extras across the family's recipes, in first-seen order."""
    out: list[str] = []
    for recipe in _family_recipes(family):
        for extra in (recipe.env or {}).get("extras", []) or []:
            if extra not in out:
                out.append(extra)
    return out


def _pip_packages(recipe: Recipe) -> list[str]:
    """Install set for a family's worker env, order-deduped.

    pyyaml (the worker imports ollada.recipes), torch, the requested
    model's transformers pin, accelerate, then the UNION of the family's
    extras (see the module docstring: per-model extras break the
    second model of a family). Extras may repeat torch/accelerate —
    dedupe keeps the first occurrence.
    """
    spec = (recipe.env or {}).get("transformers", "")
    pkgs = ["pyyaml", "torch", f"transformers{spec}", "accelerate"]
    pkgs.extend(_family_extras(recipe.family))
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

    Raises WorkerEnvError on a build failure, a pin mismatch in an
    existing env, or a missing package in an existing env (rebuild
    explicitly; see rebuild_env). _run is a test seam for the
    subprocess calls.
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
        # An env built before the union rule (or by hand) can pass the
        # pin check while missing a package another family model needs
        # — the Dell sdar env had transformers but no einops. Refuse
        # with the same rebuild pointer instead of a bare ImportError.
        dists = installed_distributions(python)
        if dists is not None:
            needed = {_norm_dist_name(p.split("[")[0].split("=")[0].split(">")[0].split("<")[0])
                      for p in _pip_packages(recipe)}
            missing = sorted(n for n in needed if n not in dists)
            if missing:
                raise WorkerEnvError(
                    f"worker env for {family} is missing {', '.join(missing)}. "
                    f"Rebuild it: ollada workers rebuild {recipe.name}")
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
    # Fresh detection at build time: never inherit a stale cached
    # result from an earlier call in this process — a transient
    # detection failure would otherwise bake the wrong torch index
    # into the env for every run after it.
    hardware = fit.detect_hardware(refresh=True)
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
        # GPU machine: torch from PyTorch's CUDA index, everything
        # else from PyPI — a single --index-url would move all
        # packages, not just torch, and plain PyPI torch is CPU-only.
        torch_pkgs = [p for p in pkgs if p == "torch"]
        rest = [p for p in pkgs if p != "torch"]
        _run_checked([str(python), "-m", "pip", "install",
                      "--index-url", CUDA_TORCH_INDEX, *torch_pkgs],
                     f"install torch (CUDA index) for {family}", _run)
        if rest:
            _run_checked([str(python), "-m", "pip", "install", *rest],
                         f"install worker packages for {family}", _run)
    # Verify the installed torch matches the detected hardware. A
    # GPU machine whose torch reports no CUDA is a broken env, not a
    # slow one — fail loudly here instead of surfacing minutes later
    # as CPU generation (the desktop dream env did exactly that).
    if hardware.gpu_name is not None and not _worker_torch_has_cuda(python, _run):
        raise WorkerEnvError(
            f"worker env for {family} was built on a GPU machine "
            f"({hardware.gpu_name}), but its torch reports no CUDA. "
            f"The torch install did not pick up CUDA — rebuild it: "
            f"ollada workers rebuild {recipe.name}")
    # Guard against incompatible family pins: pip resolved the requested
    # model's pin, but a sibling model may need a disjoint range. Fail
    # loudly at build time rather than thrashing between rebuilds.
    installed = installed_transformers(python)
    if installed is not None:
        for sibling in _family_recipes(family):
            sibling_spec = str((sibling.env or {}).get("transformers", ""))
            if sibling_spec and not spec_satisfied(sibling_spec, installed):
                raise WorkerEnvError(
                    f"worker env for {family} resolved transformers {installed}, "
                    f"which does not satisfy {sibling.name}'s pin "
                    f"transformers{sibling_spec} — this family needs "
                    f"per-model envs, which are not built yet")
    return python


def rebuild_env(recipe: Recipe, _run=subprocess.run) -> Path:
    """Delete a family's worker env and rebuild it from the recipe."""
    shutil.rmtree(venv_path(recipe.family), ignore_errors=True)
    if venv_python(recipe.family).exists():
        # Windows file locking (or a still-running worker) can leave
        # the env behind while ignore_errors hides it; proceeding
        # would make ensure_env return the old env untouched while
        # this command reports success — the desktop dream rebuild
        # did exactly that, keeping its CPU torch.
        raise WorkerEnvError(
            f"worker env for {recipe.family} could not be deleted — "
            f"it is probably still in use. Stop any running ollada "
            f"process and rebuild again: "
            f"ollada workers rebuild {recipe.name}")
    return ensure_env(recipe, _run)


def _worker_torch_has_cuda(python, _run) -> bool:
    """True only if the worker env's torch affirmatively reports CUDA."""
    try:
        proc = _run([str(python), "-c",
                     "import torch; print(torch.cuda.is_available())"],
                    capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "True"


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
