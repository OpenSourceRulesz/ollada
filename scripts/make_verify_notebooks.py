"""Generate the Colab verification notebooks in notebooks/.

One notebook per shelf model. Each notebook is a SEPARATE fresh Colab
environment, so per-family dependency pins can conflict safely (Dream/SDAR
share transformers==4.53.3; DiffCoder's custom code targets 5.3.0).

Every notebook verifies OUR code path (catalog recipe -> family adapter ->
ModelManager) on a real T4 — the Phase 1 real-load gate (PLAN.md section 7).
Re-run this script after editing the SPECS; never hand-edit the .ipynb files.
"""

import json
from pathlib import Path

DREAM_PROMPTS = [
    "In one or two sentences, what is a diffusion language model?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the steps.",
    "Write a Python function that returns the nth Fibonacci number, with a docstring.",
    "Give me three tips for staying focused while working from home, as a short list.",
]

SPECS = [
    {
        "filename": "verify_dream_t4.ipynb",
        "model": "dream:7b",
        "version": "0.1.1",
        "subject": "Dream-v0-Instruct-7B in 8-bit",
        "pin_comment": "pinned for the Dream family: transformers 4.53.3, see the recipe",
        "pkgs": ["transformers==4.53.3", "torch", "accelerate", "bitsandbytes", "pyyaml"],
        "prompt_note": "chat / reasoning / code / long-ish",
        "prompts": DREAM_PROMPTS,
        "extra_md": "",
    },
    {
        "filename": "verify_sdar_t4.ipynb",
        "model": "sdar:4b",
        "version": "0.1.0",
        "subject": "SDAR-4B-Chat (bf16 weights, fp16 load)",
        "pin_comment": "same env as Dream: transformers 4.53.3 satisfies SDAR's >=4.52.4",
        "pkgs": ["transformers==4.53.3", "torch", "accelerate", "pyyaml"],
        "prompt_note": "same 4 as dream:7b for comparability, + an instruction-following probe",
        "prompts": DREAM_PROMPTS
        + ["Reply with exactly one word: what is the capital of France?"],
        "extra_md": (
            "\nThe 5th prompt is an instruction-following probe on purpose: "
            "IFEval is SDAR's weak spot in its paper (56.6), so we test it directly.\n"
        ),
    },
    {
        "filename": "verify_diffcoder_t4.ipynb",
        "model": "diffcoder:8b",
        "version": "0.1.0",
        "subject": "Stable-DiffCoder-8B-Instruct in 8-bit",
        "pin_comment": "DiffCoder family: transformers 5.3.0 (the version its config was saved with)",
        "pkgs": ["transformers==5.3.0", "torch", "accelerate", "bitsandbytes", "pyyaml"],
        "prompt_note": "coding set: generation / bug-fix / completion / one general sanity check",
        "prompts": [
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
                "palindrome, ignoring case and non-alphanumeric characters. Only output "
                "the completed code:\n\n"
                "def is_palindrome(s):\n"
                "    # your code here"
            ),
            "In one or two sentences, what does a hash map do?",
        ],
        "extra_md": (
            "\nGeneration uses the recipe's provisional defaults (steps=128; the vendor "
            "example uses 512). Watch item: the model's block-mask code builds its mask "
            "in bfloat16 while the recipe loads the model in float16 on a T4 — if a "
            "dtype error appears, that *is* a finding; leave the cell and report it.\n"
        ),
    },
]


def cells_for(spec):
    model = spec["model"]
    version = spec["version"]
    pkgs_src = ", ".join(f'"{p}"' for p in spec["pkgs"])
    prompts_src = ",\n    ".join(json.dumps(p) for p in spec["prompts"])
    return [
        ("markdown", f"""# ollada — verify {model} on a T4 (notebook v{version})

**Before you start:** Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.

This runs ollada's real code (catalog recipe → family adapter → ModelManager) against
{spec["subject"]}. Success = the last cell prints a VERIFICATION RECORD.
If anything crashes, the failing cell *is* the bug report — leave it and tell Muse.{spec["extra_md"]}"""),
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
NOTEBOOK_VERSION = "{version}"  # also used by the record cell — keep it a variable, not a literal (dream v0.1.0 bug)
print("notebook version:", NOTEBOOK_VERSION)
print("ollada version:", ollada.__version__)
assert torch.cuda.is_available(), "no CUDA — set the runtime to T4 GPU first"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {{name}}  VRAM: {{total:.1f}} GB")"""),
        ("code", f"""# 3 — load {model} through ollada itself (timed)
import time, torch
from ollada.runtime.manager import ModelManager

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
manager = ModelManager()
adapter = manager.ensure_loaded("{model}")
load_s = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"loaded {model} in {{load_s:.1f}}s — peak VRAM {{peak:.2f}} GB")
print("ready on cuda")"""),
        ("code", f"""# 4 — fixed prompt set ({spec["prompt_note"]})
prompts = [
    {prompts_src},
]
results = []
for p in prompts:
    r = manager.chat("{model}", [{{"role": "user", "content": p}}])
    tps = r.tokens / r.seconds if r.seconds else 0
    results.append({{"prompt": p[:40], "tokens": r.tokens, "seconds": round(r.seconds, 2), "tok_per_s": round(tps, 1)}})
    print("=" * 70)
    print("PROMPT:", p)
    print(f"[{{r.tokens}} tokens in {{r.seconds:.1f}}s = {{tps:.1f}} tok/s]")
    print(r.text)"""),
        ("code", f"""# 5 — VERIFICATION RECORD (paste this whole block back to Muse)
import json, torch
record = {{
    "model": "{model}",
    "notebook_version": NOTEBOOK_VERSION,
    "ollada_version": ollada.__version__,
    "hardware": torch.cuda.get_device_name(0),
    "load_seconds": round(load_s, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "runs": results,
}}
print(json.dumps(record, indent=2))"""),
    ]


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code_cell(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def build(spec, out_dir):
    cells = cells_for(spec)
    # Regression guard for the dream v0.1.0 bug: the record cell references
    # NOTEBOOK_VERSION, so an earlier cell must define it as a variable.
    all_src = "\n".join(src for _, src in cells)
    assert "NOTEBOOK_VERSION = " in all_src, "NOTEBOOK_VERSION variable never defined"
    assert '"notebook_version": NOTEBOOK_VERSION' in all_src
    # The generation cell's sources must be valid Python once rendered.
    for kind, src in cells:
        if kind == "code":
            compile(src, f"{spec['filename']} cell", "exec")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": [md_cell(s) if kind == "markdown" else code_cell(s) for kind, s in cells],
    }
    out = out_dir / spec["filename"]
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells, notebook v{spec['version']}, model {spec['model']})")


def main():
    out_dir = Path(__file__).parent.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for spec in SPECS:
        build(spec, out_dir)


if __name__ == "__main__":
    main()
