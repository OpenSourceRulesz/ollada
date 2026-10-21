#!/usr/bin/env python3
"""Generate the steps-sweep notebooks (notebooks/sweep_{dream,diffcoder}_t4.ipynb).

Why: the recipes' step counts were provisional when the models were verified
(dream steps=32 — its truncated list / garbled code are suspect; diffcoder
steps=128 vs the vendor's 512 — its 6.7 tok/s is suspect). The sweep runs the
SAME fixed prompts as verification at several step counts and records speed
per config; quality is judged from the printed outputs. Decision rule: the
new default is the LOWEST step count whose outputs show no visible quality
loss against the highest count run.

SDAR is not swept: its knob (denoising_steps per block) has different
semantics and its verified defaults already behave.

Run: python3 scripts/make_sweep_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"

DREAM_PROMPTS = [
    "In one or two sentences, what is a diffusion language model?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the steps.",
    "Write a Python function that returns the nth Fibonacci number, with a docstring.",
    "Give me three tips for staying focused while working from home, as a short list.",
]

DIFFCODER_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number, with a docstring.",
    (
        "This function is supposed to return the average of a list of numbers, "
        "but it crashes on an empty list and has another bug. Find and fix both "
        "bugs, and show the corrected function:\n\n"
        "def average(nums):\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total =+ n\n"
        "    return total / len(nums)"
    ),
    (
        "Complete this Python function so it checks whether a string is a "
        "palindrome, ignoring case and non-alphanumeric characters. Only "
        "output the completed code:\n\n"
        "def is_palindrome(s):\n"
        "    # your code here"
    ),
    "In one or two sentences, what does a hash map do?",
]

SPECS = [
    {
        "filename": "sweep_dream_t4.ipynb",
        "model": "dream:7b",
        "version": "0.1.0",
        "pin_comment": "pinned for the Dream family: transformers 4.53.3, see the recipe",
        "pkgs": ["transformers==4.53.3", "torch", "accelerate", "bitsandbytes", "pyyaml"],
        "steps": [32, 64, 128, 256],
        "current_default": 32,
        "prompts": DREAM_PROMPTS,
        "why": (
            "The verified default is steps=32 on a 256-token canvas — and the verification "
            "run's weak spots (garbled Fibonacci, list truncated at '3.') are exactly what "
            "too few denoising steps look like. If 64+ fixes them, the default was wrong, "
            "not the model."
        ),
        "eta": "~10 min of generation after the load",
    },
    {
        "filename": "sweep_diffcoder_t4.ipynb",
        "model": "diffcoder:8b",
        "version": "0.1.0",
        "pin_comment": "DiffCoder family: transformers 5.3.0 (the version its config was saved with)",
        "pkgs": ["transformers==5.3.0", "torch", "accelerate", "bitsandbytes", "pyyaml"],
        "steps": [64, 128, 256],
        "current_default": 128,
        "prompts": DIFFCODER_PROMPTS,
        "why": (
            "The verified default is steps=128 (the vendor example uses 512) and it produced "
            "6.7 tok/s on the T4 — steps are the speed knob. Steps must divide the 64 blocks "
            "(gen_length 256 / block_length 4), hence 64/128/256. 512 is excluded: at T4 "
            "speeds it would take ~3 min per prompt and teach us nothing 256 doesn't."
        ),
        "eta": "~8 min of generation after the load",
    },
]


def cells_for(spec):
    model = spec["model"]
    version = spec["version"]
    pkgs_src = ", ".join(f'"{p}"' for p in spec["pkgs"])
    prompts_src = ",\n    ".join(json.dumps(p) for p in spec["prompts"])
    steps_src = ", ".join(str(s) for s in spec["steps"])
    return [
        ("markdown", f"""# ollada — steps sweep for {model} on a T4 (notebook v{version})

**Before you start:** Runtime → Change runtime type → **T4 GPU** (High-RAM if offered), then Runtime → Run all.

{spec["why"]}

Same fixed prompts as the verification run, several step counts, speed recorded per
config. **Decision rule:** the new recipe default is the lowest step count whose
outputs show no visible quality loss against the highest count. Send Muse the
SWEEP RECORD (last cell) and the printed outputs. Expected: {spec["eta"]}."""),
        ("code", f"""# 1 — install ({spec["pin_comment"]})
import subprocess, sys
pkgs = [
    {pkgs_src},
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "git+https://github.com/OpenSourceRulesz/ollada@main"], check=True)
print("installed")"""),
        ("code", f"""# 2 — version stamp + GPU check (acceptance gate)
import torch, ollada
NOTEBOOK_VERSION = "{version}"  # also used by the record cell — keep it a variable, not a literal
print("notebook version:", NOTEBOOK_VERSION)
print("ollada version:", ollada.__version__)
assert torch.cuda.is_available(), "no CUDA — set the runtime to T4 GPU first"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {{name}}  VRAM: {{total:.1f}} GB")"""),
        ("code", f"""# 3 — load {model} through ollada itself (timed)
import time, torch
from ollada.runtime.manager import ModelManager

# Rerun hygiene: drop stale model references from a previous run in this
# kernel before loading (a second resident copy OOMs a 16GB card).
import gc
for _stale in ("adapter", "manager"):
    globals().pop(_stale, None)
gc.collect()
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
manager = ModelManager()
adapter = manager.ensure_loaded("{model}")
load_s = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"loaded {model} in {{load_s:.1f}}s — peak VRAM {{peak:.2f}} GB")
print("ready on cuda")"""),
        ("code", f"""# 4 — the sweep: same prompts, each step count (current default: {spec["current_default"]})
prompts = [
    {prompts_src},
]
STEPS = [{steps_src}]
rows = []
for steps in STEPS:
    for p in prompts:
        r = manager.chat("{model}", [{{"role": "user", "content": p}}], overrides={{"steps": steps}})
        tps = r.tokens / r.seconds if r.seconds else 0
        rows.append({{"steps": steps, "prompt": p[:40], "tokens": r.tokens,
                      "seconds": round(r.seconds, 2), "tok_per_s": round(tps, 1)}})
        print("=" * 70)
        print(f"STEPS={{steps}}  PROMPT: {{p[:60]}}")
        print(f"[{{r.tokens}} tokens in {{r.seconds:.1f}}s = {{tps:.1f}} tok/s]")
        print(r.text)
print("SWEEP DONE")"""),
        ("code", f"""# 5 — SWEEP RECORD (paste this whole block back to Muse)
import json, torch
import ollada as _ollada
record = {{
    "kind": "steps_sweep",
    "model": "{model}",
    "notebook_version": NOTEBOOK_VERSION,
    "ollada_version": _ollada.__version__,
    "hardware": torch.cuda.get_device_name(0),
    "load_seconds": round(load_s, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "rows": rows,
}}
print(json.dumps(record, indent=2))"""),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        cells = []
        all_src = ""
        for kind, src in cells_for(spec):
            all_src += src + "\n"
            if kind == "code":
                compile(src, "<cell>", "exec")  # syntax guard, same as verify gen
                cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [],
                              "source": src.splitlines(keepends=True)})
            else:
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": src.splitlines(keepends=True)})
        assert "NOTEBOOK_VERSION = " in all_src, "NOTEBOOK_VERSION variable never defined"
        assert '"notebook_version": NOTEBOOK_VERSION' in all_src
        nb = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python", "version": "3.13"}},
              "nbformat": 4, "nbformat_minor": 0}
        path = OUT / spec["filename"]
        path.write_text(json.dumps(nb, indent=1))
        print(f"wrote {path} ({len(cells)} cells, notebook v{spec['version']}, model {spec['model']})")


if __name__ == "__main__":
    main()
