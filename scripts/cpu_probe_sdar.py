#!/usr/bin/env python3
"""CPU probe: time SDAR generation on CPU through ollada's own adapter.

Why this exists: as of 2026-10-23 there are no published CPU timings for
any small block-causal diffusion LM. This script produces them the ollada
way — the family's real adapter (with its load-time repairs), fixed
prompts from the T4 verification set, everything measured and labelled,
nothing estimated silently.

It loads ONE SDAR model on CPU and runs each fixed prompt twice: at
recipe defaults, and at a reduced-cost config (denoising_steps halved —
the adapter exposes step controls as chat overrides). Results go to
<out>.txt (human-readable) and <out>.json (structured).

The probe refuses to produce numbers it cannot stand behind:

- If CUDA is available it exits: the adapter would take the GPU branch,
  and GPU timings labelled "CPU" are worse than no timings.
- After load it asserts the CPU branch actually engaged (model on CPU,
  dtype bfloat16, the adapter's "cpu load" note present) and exits
  loudly if not.
- Forward-pass counts are recorded only if the adapter reports them
  (GenResult.nfe). SDAR's loop does not, so the output says so instead
  of estimating.

Usage (from the repo root of a feat/sdar-cpu checkout):
    python scripts/cpu_probe_sdar.py [--model JetLM/SDAR-1.7B-Chat]
        [--gen-length 128] [--out cpu_probe_results] [--threads N]
        [--no-reduced]

--model accepts a catalog name (e.g. sdar:4b) or an HF repo id (e.g.
JetLM/SDAR-1.7B-Chat). A repo id derives an UNVERIFIED probe recipe from
the sdar:4b template with the source swapped — same family defaults,
nothing inherited from the 4B's verification block.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The T4 verification prompt set (same prompts, same expected answers).
PROMPTS = [
    {
        "id": "train-speed",
        "prompt": "A train travels 45 km in 30 minutes. What is its speed in km/h? "
        "Show your reasoning briefly.",
        "expected_substring": "90",
    },
    {
        "id": "capital-one-word",
        "prompt": "What is the capital of France? Answer with exactly one word.",
        "expected_substring": "paris",
    },
    {
        "id": "fib-code-only",
        "prompt": "Write a Python function fib(n) that returns the nth Fibonacci "
        "number. Only output code.",
        "expected_substring": "def fib",
    },
]

# Known bf16 sizes for repo ids the probe is normally pointed at; used
# only for the derived recipe's metadata (fit estimates), never measured.
_KNOWN_SIZES_GB = {"JetLM/SDAR-1.7B-Chat": 3.4, "JetLM/SDAR-4B-Chat": 8.8}


def resolve_recipe(model_arg: str):
    """Catalog name -> the shelf recipe. HF repo id -> an unverified
    probe recipe derived from the sdar:4b template."""
    from ollada.recipes import get_recipe

    if "/" not in model_arg:
        return get_recipe(model_arg)
    template = get_recipe("sdar:4b")
    return dataclasses.replace(
        template,
        name="sdar:probe",
        source=model_arg,
        description=f"CPU probe recipe derived from sdar:4b for {model_arg} "
        "(unverified — probe use only)",
        size_gb_bf16=_KNOWN_SIZES_GB.get(model_arg, template.size_gb_bf16),
        verified={},
        path=None,
    )


class PeakRSS:
    """Peak resident memory of this process, in GB.

    With psutil: a sampler thread polls RSS during the run. Without it:
    point samples only, and the results say so — a degraded peak is
    labelled, never presented as a sampled one.
    """

    def __init__(self):
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None
        self.sampled = self._proc is not None
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        if self._proc is None:
            return 0
        return self._proc.memory_info().rss

    def start(self) -> None:
        self._peak = max(self._peak, self._sample())
        if self._proc is None:
            return

        def _poll():
            while not self._stop.is_set():
                self._peak = max(self._peak, self._sample())
                self._stop.wait(0.05)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def snap(self) -> None:
        self._peak = max(self._peak, self._sample())

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.snap()
        return self._peak / 1e9


def machine_info(torch) -> dict:
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine(),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    try:
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["torch_threads"] = None
    try:
        info["torch_interop_threads"] = torch.get_num_interop_threads()
    except Exception:
        info["torch_interop_threads"] = None
    try:
        import psutil

        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["psutil"] = True
    except ImportError:
        info["cpu_physical_cores"] = None
        info["cpu_logical_cores"] = os.cpu_count()
        info["ram_total_gb"] = None
        info["psutil"] = False
    return info


def _fail(message: str, results: dict | None = None) -> "SystemExit":
    print(f"\nPROBE REFUSED: {message}", file=sys.stderr)
    if results:
        print("\nPartial state at refusal:", file=sys.stderr)
        print(json.dumps(results, indent=2, default=str), file=sys.stderr)
    return SystemExit(3)


def run_probe(args) -> dict:
    try:
        import torch
    except ImportError:
        raise _fail(
            "torch is not importable in this environment. Install CPU torch "
            "first (pip install --index-url "
            "https://download.pytorch.org/whl/cpu torch)."
        )

    if torch.cuda.is_available():
        raise _fail(
            "CUDA is available on this machine. The SDAR adapter would take "
            "its GPU branch (recipe dtype, device_map='auto'), so any timings "
            "would be GPU timings. This probe measures CPU only — run it on "
            "a machine with no usable CUDA device."
        )

    if args.threads:
        torch.set_num_threads(args.threads)

    from ollada.adapters import create_adapter

    recipe = resolve_recipe(args.model)
    results: dict = {
        "probe": "ollada SDAR CPU probe",
        "model_arg": args.model,
        "recipe_name": recipe.name,
        "recipe_source": recipe.source,
        "recipe_verified": recipe.is_verified,
        "gen_length": args.gen_length,
        "machine": machine_info(torch),
        "notes": [],
    }

    adapter = create_adapter(recipe)
    peak = PeakRSS()
    if not peak.sampled:
        results["notes"].append(
            "psutil not installed — peak memory is the max of point samples, "
            "not a polled peak"
        )
    peak.start()
    started = time.perf_counter()
    adapter.load()
    results["load_seconds"] = round(time.perf_counter() - started, 2)
    results["load_notes"] = list(getattr(adapter, "load_notes", []))
    peak.snap()

    # "You gotta be sure": the CPU branch must have actually engaged.
    try:
        device = str(adapter.model.device)
    except Exception:
        device = str(next(adapter.model.parameters()).device)
    results["model_device"] = device
    results["model_dtype"] = str(adapter.model.dtype)
    if device != "cpu":
        raise _fail(f"model loaded on device '{device}', not CPU", results)
    if adapter.model.dtype != torch.bfloat16:
        raise _fail(
            f"model dtype is {adapter.model.dtype}, not bfloat16 — the CPU "
            "load branch (_resolve_load_plan) did not engage",
            results,
        )
    if not any(n.startswith("cpu load") for n in results["load_notes"]):
        raise _fail(
            "adapter load_notes carry no 'cpu load' entry — cannot confirm "
            "the CPU branch engaged",
            results,
        )

    configs = [
        ("recipe-defaults", {"gen_length": args.gen_length}),
    ]
    if not args.no_reduced:
        configs.append(
            (
                "reduced (denoising_steps=2, recipe default is 4)",
                {"gen_length": args.gen_length, "denoising_steps": 2},
            )
        )

    runs = []
    nfe_reported = False
    for config_label, overrides in configs:
        for spec in PROMPTS:
            t0 = time.perf_counter()
            result = adapter.chat(
                [{"role": "user", "content": spec["prompt"]}], overrides=overrides
            )
            wall = time.perf_counter() - t0
            peak.snap()
            if result.nfe is not None:
                nfe_reported = True
            runs.append(
                {
                    "config": config_label,
                    "prompt_id": spec["id"],
                    "prompt": spec["prompt"],
                    "wall_seconds": round(wall, 2),
                    "adapter_seconds": round(result.seconds, 2),
                    "tokens": result.tokens,
                    "forward_passes": result.nfe,
                    "contains_expected": spec["expected_substring"].lower()
                    in result.text.lower(),
                    "text": result.text,
                }
            )
            print(f"  [{config_label}] {spec['id']}: {wall:.1f}s")
    results["runs"] = runs
    if not nfe_reported:
        results["notes"].append(
            "forward_passes is null: the SDAR adapter does not report a "
            "forward-pass count (GenResult.nfe stays None), and this probe "
            "does not estimate one"
        )
    results["peak_rss_gb"] = round(peak.stop(), 2)
    return results


def render_txt(results: dict) -> str:
    m = results["machine"]
    lines = [
        "ollada SDAR CPU probe",
        "=" * 60,
        f"model (arg):     {results['model_arg']}",
        f"recipe:          {results['recipe_name']} <- {results['recipe_source']}"
        f" (verified: {results['recipe_verified']})",
        f"gen_length:      {results['gen_length']}",
        "",
        "machine",
        f"  os:            {m['os']}",
        f"  cpu:           {m['cpu_model']}",
        f"  cores:         physical={m['cpu_physical_cores']} logical={m['cpu_logical_cores']}",
        f"  ram total:     {m['ram_total_gb']} GB",
        f"  python:        {m['python']}",
        f"  torch:         {m['torch_version']} (threads={m['torch_threads']}, "
        f"interop={m['torch_interop_threads']})",
        "",
        f"load:            {results['load_seconds']}s, model on {results['model_device']}"
        f" at {results['model_dtype']}",
        f"peak RSS:        {results['peak_rss_gb']} GB",
        "load notes (adapter):",
    ]
    lines += [f"  - {n}" for n in results["load_notes"]]
    lines += ["", "runs"]
    for r in results["runs"]:
        lines += [
            f"  [{r['config']}] {r['prompt_id']}: wall {r['wall_seconds']}s, "
            f"tokens {r['tokens']}, forward passes {r['forward_passes']}, "
            f"expected-answer substring: {'yes' if r['contains_expected'] else 'NO'}",
            f"    text: {r['text'][:400]}",
        ]
    if results["notes"]:
        lines += ["", "notes"]
        lines += [f"  - {n}" for n in results["notes"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Time SDAR generation on CPU through ollada's adapter."
    )
    parser.add_argument(
        "--model",
        default="JetLM/SDAR-1.7B-Chat",
        help="catalog name (sdar:4b) or HF repo id (default: %(default)s)",
    )
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument(
        "--out",
        default="cpu_probe_results",
        help="prefix for the results files (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch intra-op threads (default: torch's own default)",
    )
    parser.add_argument(
        "--no-reduced",
        action="store_true",
        help="skip the reduced-cost config (denoising_steps=2) run",
    )
    args = parser.parse_args(argv)

    print(f"SDAR CPU probe — model: {args.model}, gen_length: {args.gen_length}")
    results = run_probe(args)

    txt_path = Path(f"{args.out}.txt")
    json_path = Path(f"{args.out}.json")
    txt_path.write_text(render_txt(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\nsummary")
    print(f"  load: {results['load_seconds']}s on {results['model_device']}"
          f" ({results['model_dtype']}), peak RSS {results['peak_rss_gb']} GB")
    for r in results["runs"]:
        print(f"  [{r['config']}] {r['prompt_id']}: {r['wall_seconds']}s")
    print(f"  wrote {txt_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
