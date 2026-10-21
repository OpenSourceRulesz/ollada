# ollada

**The shelf for open-source diffusion language models.**

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

**0.0.1 — name-claim / planning release.** The full plan (product definition, recipe format, initial 3-model shelf, verification gate, phased timeline) is in [PLAN.md](PLAN.md). No models are served yet; the runtime is in development.

Named the way Ollama was: Ollama ← LLaMA (the first big open autoregressive model). **Ollada ← LLaDA** (the first big open diffusion language model, Feb 2025).

## Principles

- $0 project: no telemetry, no phone-home, no paid services.
- GPU-first and honest about it (diffusion on CPU is minutes per answer).
- Vendor benchmarks are never presented as our measurements.

## License

MIT
