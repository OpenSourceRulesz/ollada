<p align="center">
  <img src="docs/assets/logo-badge.png" width="150" alt="ollada — the pig. An ollada is a Catalan pork stew; Ollama has the llama, ollada has the pig.">
</p>

<h1 align="center">ollada</h1>

<p align="center"><strong>The shelf for open-source diffusion language models.</strong></p>

Ollama is where people go to get open-source autoregressive models: clean names, one command, it just works. Diffusion language models (dLLMs) — LLaDA, Dream, SDAR, Stable-DiffCoder — have none of that. They're scattered across Hugging Face with incompatible generation code, undocumented defaults, and no verified "this actually runs" guarantee.

**ollada** is building the equivalent for diffusion:

```bash
pip install ollada
ollada pull diffcoder:8b
ollada run diffcoder:8b
ollada serve   # local API on http://localhost:11435
```

- **Curated library, short names** — `dream:7b`, `sdar:4b`, `diffcoder:8b`. A model is on the shelf only after a real-load verification run on real GPU hardware.
- **Verified recipes, not homework** — the steps / block size / threshold / quantization that actually work are baked into each model's recipe, tested and published with the hardware they were measured on.
- **A local API tools can point at** — Ollama-shaped and OpenAI-shaped endpoints.

## Status

**In development (Phase 1).** The runtime core is built: recipe catalog, family adapters (Dream / SDAR / DiffCoder; Nemotron dormant), a one-model-at-a-time manager, Ollama-shaped + OpenAI-shaped APIs, and the `list / show / pull / run / serve / ps` CLI.

**Verified shelf (real-load runs, our numbers — see [PLAN.md](PLAN.md) §7):**
- ✅ `dream:7b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 9.80GB, ~6.7 canvas tok/s at the swept recipe defaults (steps=128; the original 32-step default was faster but garbled code output). Quality notes live in its recipe.
- ✅ `sdar:4b` — Tesla T4 (Colab, High-RAM runtime), 2026-10-21: peak VRAM 8.87GB, ~12 canvas tok/s at the recipe defaults, clean quality across the fixed prompt set (correct math, runnable code, complete lists, exact one-word instruction following). Verified with ollada's load repairs active — its upstream packaging needs five of them; see the adapter.
- ✅ `diffcoder:8b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 15.15GB (the T4 is this recipe's floor — ~0.5GB headroom), ~6.7 canvas tok/s at the recipe defaults, code quality strong across the coding prompt set. Verified with ollada's mask-dtype repair active; see the recipe notes.

**The v0.1 shelf is fully verified on T4** — every model above has passed a real-load run through ollada's own code on real hardware, with the numbers recorded in its recipe.

The first shelf: `dream:7b` (general flagship) · `sdar:4b` (general, strongest per-size) · `diffcoder:8b` (coding). First planned addition after v0.1: `nemotron:3b` (NVIDIA) — its adapter is already in the tree; it joins the shelf together with the isolated-worker build its `transformers>=5` requirement forces (see PLAN.md §5 decision record).

Named the way Ollama was: Ollama ← LLaMA (the first big open autoregressive model). **Ollada ← LLaDA** (the first big open diffusion language model, Feb 2025).

## Principles

- $0 project: no telemetry, no phone-home, no paid services.
- GPU-first and honest about it (diffusion on CPU is minutes per answer).
- Vendor benchmarks are never presented as our measurements.

## License

MIT
