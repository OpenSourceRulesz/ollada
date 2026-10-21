"""Generate notebooks/verify_dream_t4.ipynb.

The notebook verifies OUR code path (ollada's ModelManager + DreamAdapter)
on a real Colab T4 — the Phase 1 real-load gate (PLAN.md section 7).
Re-run this script after editing the cell sources; never hand-edit the .ipynb.
"""

import json
from pathlib import Path

NOTEBOOK_VERSION = "0.1.1"

CELLS = [
    ("markdown", f"""# ollada — verify dream:7b on a T4 (notebook v{NOTEBOOK_VERSION})

**Before you start:** Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.

This runs ollada's real code (catalog recipe → DreamAdapter → ModelManager) against
Dream-v0-Instruct-7B in 8-bit. Success = the last cell prints a VERIFICATION RECORD.
If anything crashes, the failing cell *is* the bug report — leave it and tell Muse."""),
    ("code", """# 1 — install (pinned for the Dream family: transformers 4.53.3, see the recipe)
import subprocess, sys
pkgs = [
    "transformers==4.53.3", "torch", "accelerate", "bitsandbytes", "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "git+https://github.com/OpenSourceRulesz/ollada@main"], check=True)
print("installed")"""),
    ("code", f"""# 2 — version stamp + GPU check (acceptance gate)
import torch, ollada
NOTEBOOK_VERSION = "{NOTEBOOK_VERSION}"  # also used by the record cell — keep it a variable, not a literal (v0.1.0 bug)
print("notebook version:", NOTEBOOK_VERSION)
print("ollada version:", ollada.__version__)
assert torch.cuda.is_available(), "no CUDA — set the runtime to T4 GPU first"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {{name}}  VRAM: {{total:.1f}} GB")"""),
    ("code", """# 3 — load dream:7b through ollada itself (timed)
import time, torch
from ollada.runtime.manager import ModelManager

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
manager = ModelManager()
adapter = manager.ensure_loaded("dream:7b")
load_s = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"loaded dream:7b in {load_s:.1f}s — peak VRAM {peak:.2f} GB")
print("ready on cuda")"""),
    ("code", """# 4 — fixed prompt set (chat / reasoning / code / long-ish)
prompts = [
    "In one or two sentences, what is a diffusion language model?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the steps.",
    "Write a Python function that returns the nth Fibonacci number, with a docstring.",
    "Give me three tips for staying focused while working from home, as a short list.",
]
results = []
for p in prompts:
    r = manager.chat("dream:7b", [{"role": "user", "content": p}])
    tps = r.tokens / r.seconds if r.seconds else 0
    results.append({"prompt": p[:40], "tokens": r.tokens, "seconds": round(r.seconds, 2), "tok_per_s": round(tps, 1)})
    print("=" * 70)
    print("PROMPT:", p)
    print(f"[{r.tokens} tokens in {r.seconds:.1f}s = {tps:.1f} tok/s]")
    print(r.text)"""),
    ("code", """# 5 — VERIFICATION RECORD (paste this whole block back to Muse)
import json, torch
record = {
    "model": "dream:7b",
    "notebook_version": NOTEBOOK_VERSION,
    "ollada_version": ollada.__version__,
    "hardware": torch.cuda.get_device_name(0),
    "load_seconds": round(load_s, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "runs": results,
}
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


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": [md_cell(s) if kind == "markdown" else code_cell(s) for kind, s in CELLS],
    }
    # Regression guard for the v0.1.0 bug: the record cell references
    # NOTEBOOK_VERSION, so an earlier cell must define it as a variable.
    all_src = "\n".join(src for _, src in CELLS)
    assert "NOTEBOOK_VERSION = " in all_src, "NOTEBOOK_VERSION variable never defined"
    assert '"notebook_version": NOTEBOOK_VERSION' in all_src
    out = Path(__file__).parent.parent / "notebooks" / "verify_dream_t4.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells, notebook v{NOTEBOOK_VERSION})")


if __name__ == "__main__":
    main()
