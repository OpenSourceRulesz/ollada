# Ollada library

The shelf: every model Ollada has verified through its own code. Numbers are
measured on the hardware named — no vendor benchmarks.

| Model | Task | Weights (bf16) | Peak VRAM, measured | Speed (canvas tok/s) | Default quant | Verified | License |
|---|---|---|---|---|---|---|---|
| `diffcoder:8b` | coding | 16.0 GB | 15.15 GB | 4.1 | int8 | 2026-10-21 | MIT |
| `dream:7b` | chat | 15.2 GB | 9.8 GB | 6.7 | int8 | 2026-10-21 | Apache-2.0 |
| `sdar:1.7b` | chat | 4.1 GB | — (CPU) | 6.1 | bf16 | 2026-10-23 | Apache-2.0 |
| `sdar:4b` | chat | 8.8 GB | 8.87 GB | 12.3 | bf16 | 2026-10-21 | Apache-2.0 |

GPU numbers are from a Colab Tesla T4 (2026-10-21) and are the conservative
floor; desktop verification (RTX 5060 Ti) is the pre-launch re-check.
`sdar:1.7b` was verified on a laptop CPU (2026-10-23) — no GPU was used, so
it has no VRAM figure. Weights come from Hugging Face — Ollada hosts no
registry.

Full recipes and verification notes live on each model's site page
(https://opensourcerulesz.github.io/ollada/models/).
