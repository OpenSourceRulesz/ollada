<p align="center">
  <img src="docs/assets/logo.svg" width="120" alt="Ollada logo — scattered dots settling into order: diffusion, denoised.">
</p>

<h1 align="center">Ollada</h1>

<p align="center"><strong>The shelf for open-source diffusion language models.</strong></p>

Ollama is where people go to get open-source autoregressive models: clean names, one command, it just works. Diffusion language models (dLLMs) — LLaDA, Dream, SDAR, Stable-DiffCoder — have none of that. They're scattered across Hugging Face with incompatible generation code, undocumented defaults, and no verified "this actually runs" guarantee.

**Ollada** is building the equivalent for diffusion:

```bash
pip install "ollada[cpu]"   # laptop, no GPU needed
ollada pull sdar:1.7b
ollada run sdar:1.7b
```

On a GPU machine, `pip install "ollada[gpu]"` instead (adds bitsandbytes and einops for the GPU shelf), then `ollada pull diffcoder:8b` / `ollada run diffcoder:8b`. `ollada serve` needs `pip install "ollada[server]"` — the local API on http://localhost:11435. Bare `pip install ollada` gives you only the shelf commands (`list` / `show` / `recommend`); it can't load a model. Quotes around the brackets are for PowerShell; on bash/zsh they're optional. CPU-only machine? Install torch from the CPU index first — `pip install torch --index-url https://download.pytorch.org/whl/cpu` — and pip will reuse it instead of downloading the ~2GB CUDA bundle.

- **Curated library, short names** — `dream:7b`, `sdar:4b`, `diffcoder:8b`, `sdar:1.7b`. A model is on the shelf only after a real-load verification run through Ollada's own code on named hardware — a real GPU for the flagship shelf; `sdar:1.7b` is the first CPU-verified entry (an ordinary laptop, no GPU).
- **Verified recipes, not homework** — the steps / block size / threshold / quantization that actually work are baked into each model's recipe, tested and published with the hardware they were measured on.
- **A fit check that knows the numbers** — `ollada recommend` compares each model's measured peak VRAM against your GPU and says fits / tight / won't-fit *before* you download 15GB; a load that can't fit is refused (override with `--force`; the API answers 409, or pass `options: {"force": true}`). Every load and generation also records its peak locally — nothing leaves your machine — so over time `ollada show` quotes measurements from *your* GPU, not the T4 ours came from.
- **A local API tools can point at** — Ollama-shaped and OpenAI-shaped endpoints.
- **A run experience built for diffusion** — loads are quiet (no library progress bars or warning spam); on a terminal, SDAR runs show the canvas itself denoising — committed words, the model's current guesses, untouched masks — and `--length` (or `/length` in the REPL) sizes the canvas to the ask, because a diffusion canvas is fixed before generation and its cost scales with its size.

## Status

**In development (Phase 1).** The runtime core is built: recipe catalog, family adapters (Dream / SDAR / DiffCoder; Nemotron dormant), a one-model-at-a-time manager, Ollama-shaped + OpenAI-shaped APIs, and the `list / show / pull / run / serve / ps` CLI.

**Verified shelf (real-load runs, our numbers — see [PLAN.md](PLAN.md) §7):**
- ✅ `dream:7b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 9.80GB, ~6.7 canvas tok/s at the swept recipe defaults (steps=128; the original 32-step default was faster but garbled code output). Quality notes live in its recipe.
- ✅ `sdar:4b` — Tesla T4 (Colab, High-RAM runtime), 2026-10-21: peak VRAM 8.87GB, ~12 canvas tok/s at the recipe defaults, clean quality across the fixed prompt set (correct math, runnable code, complete lists, exact one-word instruction following). Verified with Ollada's load repairs active — its upstream packaging needs five of them; see the adapter. Also verified on CPU (laptop, 2026-10-23 — see FINDINGS §7).
- ✅ `sdar:1.7b` — Intel Core Ultra 7 365 laptop (32GB RAM, Windows 11), CPU only, 2026-10-23: peak host RAM 3.9GB, ~6.1 canvas tok/s on full-length answers at recipe defaults (short answers finish in seconds via early stop), all fixed prompts correct. The shelf's CPU pick; a GPU verification has not been run for this model.
- ✅ `diffcoder:8b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 15.15GB (the T4 is this recipe's floor — ~0.5GB headroom), ~6.7 canvas tok/s at the recipe defaults, code quality strong across the coding prompt set. Verified with Ollada's mask-dtype repair active; see the recipe notes.

**The v0.1 flagship shelf is fully verified on T4**, and the shelf has since gained its first CPU-verified model — every model above has passed a real-load run through Ollada's own code on the hardware named in its recipe, with the numbers recorded there.

Field notes from that work — the measured numbers, the steps sweeps, the upstream defect catalog, and which vendor benchmarks nobody has reproduced — live in [docs/FINDINGS.md](docs/FINDINGS.md) (early; desktop re-verification pending). The shelf side-by-side is [docs/LIBRARY.md](docs/LIBRARY.md).

The first shelf: `dream:7b` (general flagship) · `sdar:4b` (general, strongest per-size) · `diffcoder:8b` (coding) · `sdar:1.7b` (the CPU pick). First planned addition after v0.1: `nemotron:3b` (NVIDIA) — its adapter is already in the tree; it joins the shelf together with the isolated-worker build its `transformers>=5` requirement forces (see PLAN.md §5 decision record).

Named the way Ollama was: Ollama ← LLaMA (the first big open autoregressive model). **Ollada ← LLaDA** (the first big open diffusion language model, Feb 2025).

## Principles

- $0 project: no telemetry, no phone-home, no paid services.
- GPU-first for the flagship shelf, and honest about hardware: the SDAR models also run on CPU — measured on a laptop at seconds for short answers (1.7B) to minutes for long ones (4B). Dream and DiffCoder remain GPU models; nothing here pretends otherwise.
- Vendor benchmarks are never presented as our measurements.

## License

MIT
