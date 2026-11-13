[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ollada?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ollada)

<p align="center">
  <img src="docs/assets/logo.svg" width="120" alt="Ollada logo — scattered dots settling into order: diffusion, denoised.">
</p>

<h1 align="center">Ollada</h1>

<p align="center"><strong>The shelf for open-source diffusion language models.</strong></p>

<p align="center"><a href="https://opensourcerulesz.github.io/ollada">Live site</a> — the shelf, verified numbers, and install guide.</p>

Ollama is where people go to get open-source autoregressive models: clean names, one command, it just works. Diffusion language models (dLLMs) — LLaDA, Dream, SDAR, Stable-DiffCoder — have none of that. They're scattered across Hugging Face with incompatible generation code, undocumented defaults, and no verified "this actually runs" guarantee.

**Ollada** is building the equivalent for diffusion:

```bash
pip install ollada
ollada pull sdar:1.7b
ollada run sdar:1.7b
```

Needs Python 3.10 or newer — pip is the installer. The install ships a bundled Rust binary that runs the CLI, the server, and the worker setup, so there are no extras to pick: no `[gpu]` or `[server]` needed to run models or serve. (The extras still exist for direct Python use — probes, notebooks.)

Models run in per-family worker environments, built on first load of a family (one venv per family, pinned from the recipe, under `~/.ollada/workers/` — `ollada workers` lists them, `ollada workers rebuild <model>` rebuilds one). That's how dream's transformers 4.x and diffcoder's 5.3.0 coexist. The first load of a family downloads torch, so it's slow once. On a machine with no NVIDIA GPU, the worker env pulls torch from the CPU index automatically instead of the ~2GB CUDA bundle.

- **Curated library, short names** — `diffcoder:8b`, `diffcoder:8b4bit`, `dream:7b`, `sdar:4b`, `nemotron:3b`, `sdar:1.7b`. A model is on the shelf only after a real-load verification run through Ollada's own code on named hardware — a real GPU for the flagship shelf; `sdar:1.7b` is the first CPU-verified entry and `nemotron:3b` the second (ordinary laptops, no GPU).
- **Verified recipes, not homework** — the steps / block size / threshold / quantization that actually work are baked into each model's recipe, tested and published with the hardware they were measured on.
- **A fit check that knows the numbers** — `ollada recommend` compares each model's measured peak VRAM against your GPU and says fits / tight / won't-fit *before* you download 15GB; a load that can't fit is refused (override with `--force`; the API answers 409, or pass `options: {"force": true}`). Every load and generation also records its peak locally — nothing leaves your machine — so over time `ollada show` quotes measurements from *your* GPU, not the T4 ours came from.
- **A local API tools can point at** — Ollama-shaped and OpenAI-shaped endpoints.
- **A run experience built for diffusion** — loads are quiet (no library progress bars or warning spam); on a terminal, SDAR runs show the canvas itself denoising — committed words, the model's current guesses, untouched masks — and `--length` (or `/length` in the REPL) sizes the canvas to the ask, because a diffusion canvas is fixed before generation and its cost scales with its size.

## Status

**Version 0.3.7 shipped 2026-11-12.** The runtime core is built: recipe catalog, family adapters (Dream / SDAR / DiffCoder / Nemotron), a one-model-at-a-time manager with per-family isolated worker environments, Ollama-shaped + OpenAI-shaped APIs, and the `list / show / pull / run / serve / ps / workers` CLI. Since 0.3.0: 0.3.5 added the verified `diffcoder:8b4bit` recipe; 0.3.6 added the first outside contribution (GB10 detection fix), the Colab worker-venv fix, and `POST /v1/completions`; 0.3.7 rewrote the front-end in Rust — `pip install ollada` now bundles a single binary (Linux / macOS arm64 / Windows wheels) that runs the CLI, the server, and the worker setup, proven on a real Windows laptop (Dell, no GPU: `run sdar:1.7b` answered Paris in 8.8s from a cold start, serve healthy, worker rebuild through Rust).

**Verified shelf (real-load runs, our numbers — see [PLAN.md](PLAN.md) §7):**
- ✅ `diffcoder:8b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 15.15GB (the T4 is this recipe's floor — ~0.5GB headroom), ~4.1 canvas tok/s at the recipe defaults (steps=256), code quality strong across the coding prompt set. Also verified on the desktop RTX 5060 Ti, 2026-10-25: peak 13.65 GB (see FINDINGS §8). Verified with Ollada's mask-dtype repair active; see the recipe notes.
- ✅ `diffcoder:8b4bit` — NVIDIA A100-SXM4-80GB (Colab), 2026-11-06: same model in 4-bit (NF4 weight-only), peak VRAM 6.59 GB, fixed prompt set 3/3 through `ollada run` (train 90 km/h in 35.4s, capital Paris in 18.7s, fib correct in 26.2s). Fits an 8 GB-class card, with headroom on a 16 GB card. Desktop 5060 Ti verification still pending. Independent 4-bit quality numbers (standalone Colab, not through ollada): HumanEval full 164 at 81.1%, HumanEval+ first 50 at 88.0% — see the recipe notes.
- ✅ `dream:7b` — Tesla T4 (Colab), 2026-10-21: peak VRAM 9.80GB, ~6.7 canvas tok/s at the swept recipe defaults (steps=128; the original 32-step default was faster but garbled code output). Quality notes live in its recipe. Also verified on the desktop RTX 5060 Ti, 2026-10-25: peak 9.13 GB (see FINDINGS §8).
- ✅ `sdar:4b` — Tesla T4 (Colab, High-RAM runtime), 2026-10-21: peak VRAM 8.87GB, ~12 canvas tok/s at the recipe defaults, clean quality across the fixed prompt set (correct math, runnable code, complete lists, exact one-word instruction following). Verified with Ollada's load repairs active — its upstream packaging needs five of them; see the adapter. Also verified on CPU (laptop, 2026-10-23 — see FINDINGS §7) and on the desktop RTX 5060 Ti, 2026-10-25: peak 8.34 GB (see FINDINGS §8).
- ✅ `nemotron:3b` — Intel Core Ultra 7 365 laptop (32GB RAM, Windows 11), CPU only, 2026-10-26: fixed prompt set all correct (capital exactly "Paris", 17.9s; train 90 km/h with clean working, ~60s; Fibonacci iterative and correct, 19.0s — times include cold load per run); mini code eval 11/12 (the miss: count_vowels produced no callable function; the one task both SDARs failed, merge_dicts_sum, passed). The shelf's NVIDIA entry; a GPU verification has not been run for this model.
- ✅ `sdar:1.7b` — Intel Core Ultra 7 365 laptop (32GB RAM, Windows 11), CPU only, 2026-10-23: peak host RAM 3.9GB, ~6.1 canvas tok/s on full-length answers at recipe defaults (short answers finish in seconds via early stop), all fixed prompts correct. The shelf's CPU pick; a GPU verification has not been run for this model.

**The v0.1 flagship shelf is fully verified on T4**, and the shelf has since gained two CPU-verified models — every model above has passed a real-load run through Ollada's own code on the hardware named in its recipe, with the numbers recorded there.

Field notes from that work — the measured numbers, the steps sweeps, the upstream defect catalog, and which vendor benchmarks nobody has reproduced — live in [docs/FINDINGS.md](docs/FINDINGS.md). The shelf side-by-side is [docs/LIBRARY.md](docs/LIBRARY.md).

The shelf: `diffcoder:8b` (coding) · `diffcoder:8b4bit` (coding, 4-bit, fits 8 GB cards) · `dream:7b` (general flagship) · `sdar:4b` (general, strongest per-size) · `nemotron:3b` (NVIDIA tri-mode, CPU-verified) · `sdar:1.7b` (the CPU pick).

Named the way Ollama was: Ollama ← LLaMA (the first big open autoregressive model). **Ollada ← LLaDA** (the first big open diffusion language model, Feb 2025).

## Principles

- $0 project: no telemetry, no phone-home, no paid services.
- GPU-first for the flagship shelf, and honest about hardware: the SDAR models also run on CPU — measured on a laptop at seconds for short answers (1.7B) to minutes for long ones (4B). Dream and DiffCoder remain GPU models; nothing here pretends otherwise.
- Vendor benchmarks are never presented as our measurements.

## License

MIT
