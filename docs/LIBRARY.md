# Ollada library

The shelf: every model Ollada has verified through its own code. Numbers are
measured on the hardware named — no vendor benchmarks.

| Model | Task | Weights (bf16) | Peak VRAM, measured | Speed (canvas tok/s) | Default quant | Verified | License |
|---|---|---|---|---|---|---|---|
| `diffcoder:8b` | coding | 16.0 GB | 13.65 GB | 4.1 | int8 | 2026-10-25 | MIT |
| `dream:7b` | chat | 15.2 GB | 9.13 GB | 6.7 | int8 | 2026-10-25 | Apache-2.0 |
| `sdar:4b` | chat | 8.8 GB | 8.34 GB | 12.3 | bf16 | 2026-10-25 | Apache-2.0 |
| `nemotron:3b` | chat | 7.7 GB | — (CPU) | — | bf16 | 2026-10-26 | NVIDIA Nemotron Open Model License |
| `sdar:1.7b` | chat | 4.1 GB | — (CPU) | 6.1 | bf16 | 2026-10-23 | Apache-2.0 |

GPU peaks are from the desktop RTX 5060 Ti re-verification
(2026-10-25); first verified on a Colab Tesla T4 (2026-10-21:
diffcoder:8b 15.15 GB, dream:7b 9.80 GB, sdar:4b 8.87 GB). Speeds are
T4 canvas tok/s — desktop runs recorded times, not token counts.
Desktop notes live in FINDINGS §8 and on the main site.
`sdar:1.7b` was verified on a laptop CPU (2026-10-23) — no GPU was used, so
it has no VRAM figure. Weights come from Hugging Face — Ollada hosts no
registry.

Full recipes and verification notes live in docs/FINDINGS.md and on the
main site (https://opensourcerulesz.github.io/ollada).
