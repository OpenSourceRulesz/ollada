# Findings — field notes from putting diffusion LLMs on a shelf

**Status: early.** Everything below comes from Ollada's own real-load runs
through Ollada's own code on a Colab **Tesla T4** (2026-10-21), plus a
literature pass the same week. Desktop re-verification on an
**RTX 5060 Ti 16GB** — the machine this shelf is actually aimed at — is
still pending and is a pre-launch gate. A CPU addendum (§7,
2026-10-23) follows the T4 work — SDAR measured running on a laptop
CPU — and changes nothing about that gate. Treat the T4 numbers as a
conservative floor, not the final word: speeds will change on newer
hardware, and one fit question (DiffCoder, §6) only the desktop can
answer. Where a number is a vendor's, it is labelled a vendor's. Vendor
benchmarks are never presented as our measurements.

## 1. The shelf, measured

| Model | Peak VRAM | Speed at defaults | Load | Quality, briefly |
|---|---|---|---|---|
| `dream:7b` | 9.80 GB | ~6.7 canvas tok/s (steps=128) | 249.1s incl. first ~15GB download | math correct; Fibonacci code clean at 128 steps (garbled at 32/64) |
| `sdar:4b` | 8.87 GB | ~12 canvas tok/s | 45.1s | beat Dream on the shared set: math cleaner, Fibonacci correct and runnable, lists complete, one-word probe answered exactly |
| `diffcoder:8b` | 15.15 GB | ~4.1 tok/s (steps=256) | 89.5s | bug-fix prompt excellent (both planted bugs found, fix correct); code clean, surrounding prose is where diffusion artifacts show |

Defaults in the recipes are the swept defaults (§2), not the values the
models were first verified at.

## 2. Steps sweeps: more steps is not better

Each model was run at several step counts on a fixed prompt set
(one run per configuration, temperature 0.2 — noisy samples, not laws;
the pre-registered rule was "lowest count with no visible quality loss
versus the highest").

- **DiffCoder 64 steps collapses**: repetition degeneration
  ("by by by…" on the bug-fix prompt, "maps maps maps" on the hash-map
  prompt), early-EOS never fires — ~10 tok/s of garbage. 128 is
  code-correct but its explanation prose carries grammar artifacts
  ("accumulatinging"). **256 is the default**: same correct code, prose
  clean, 4.1 tok/s.
- **Dream is non-monotonic.** 32 steps (28 tok/s) garbled Fibonacci code
  in 2 of 2 runs; 64 (13.5 tok/s) still broke it (unterminated string,
  tuple bugs) despite the best-structured math of the sweep; **128
  (6.7 tok/s) is the default** — Fibonacci clean, no observed failures;
  256 (3.4 tok/s) produced the sweep's only *wrong* math answer
  ("45/60 = 60 hours") at half of 128's speed.
- At every step count, DiffCoder's verbose style exhausts the 256-token
  canvas mid-answer on longer prompts — a `gen_length` question, not a
  steps question. Filed as future sweep material.

## 3. Upstream defect catalog

Getting three "just load it from Hugging Face" models to actually load
took eight repairs. The CPU/Windows path added four more SDAR repairs (§7). Each was reproduced against the real code and
weights before it was fixed.

**SDAR (`JetLM/SDAR-4B-Chat`) — five defects**, all confirmed still
present in the upstream modelling file as of the 2026-10-21 research
pass (file updated Feb 2026). SDAR's HF repo has **zero discussions** —
as far as we can find, the repairs in `ollada/adapters/sdar.py` are the
only public workaround.

1. Its RMSNorm forward hard-requires `flash_attn`, whose pure-torch
   fallback is commented out upstream. → Adapter registers a stand-in
   module providing a bit-exact torch RMSNorm.
2. The HF repo is missing `fused_linear_diffusion_cross_entropy.py`,
   which its modelling imports (a dead import). → Adapter loads from a
   local overlay of the snapshot plus the authors' real file, taken
   from their GitHub `training/` folder.
3. The README's `transformers>=4.52.4` is stale: the modelling imports
   `TransformersKwargs`, which needs **≥4.57**. → Pin corrected.
4. Attention accepts only flex-attention BlockMasks, and no shipped
   code path builds them — the authors' own `generate.py` cannot run
   as shipped. → The adapter's vendored loop builds BlockMasks and
   swaps the compiled flex wrapper for eager attention.
5. `SDARForCausalLM.forward` accepts `past_key_values` but never passes
   it to the inner model, so every decode restarts with a fresh cache.
   → The loop drives `model.model` + `lm_head` directly.

**DiffCoder (`ByteDance-Seed/Stable-DiffCoder-8B-Instruct`) — two.**

1. Its `generate_block` hardcodes a **bfloat16** block mask while the
   recipe loads fp16 on T4; transformers 5.x SDPA rejects the mismatch.
   → The adapter wraps the mask builder so the mask follows the model
   dtype.
2. Card-vs-code contradiction on the confidence threshold (card: 0.95;
   vendor example: `None`). The adapter passes `threshold=None` and
   `eos_id` explicitly — the defaults silently disable early-EOS.

**Dream (`Dream-org/Dream-v0-Instruct-7B`) — ecosystem notes.** The HF
repo has been frozen since Jul 2025 and pins transformers 4.46.2.
Upstream issue #83 (open, unanswered) reports the 5.x RoPE breakage
(`KeyError: 'default'`); issue #84 reports GSM8K dropping from ~79% to
~40% at temperature 0.0 vs 0.1 — worth knowing before you "fix" its
sampling defaults. Development energy has moved to successors
(Dream-Coder-v0, DreamOn-v0, and others) rather than a Dream v1.

**One of ours.** The verification gate also caught an Ollada bug:
transformers 5.x changed `BatchEncoding` from a dict subclass to a
`UserDict`, so an `isinstance(..., dict)` unwrap silently fed the whole
batch object into `generate()`. Fixed with a `Mapping` check plus a
regression test. The gate is not theatre; it has caught us too.

## 4. Benchmark provenance

- **DiffCoder's headline numbers** (HumanEval 86.6 / MBPP 85.7) are
  **author-only**: the 2026-10-21 research pass found zero independent
  reproductions. They may be right; nobody else has shown it. Our
  recipe repeats them only with that label attached.
- **SDAR's card is partly contradicted.** Independent papers (S2D2,
  LoPA) put MBPP at 52–57 against the card's 65.4, and IFEval at 51
  against 56.6; GSM8K is roughly confirmed. SGLang now documents SDAR
  support — naming the 8B and 30B, not the 4B.
- **Dream's** card numbers are the authors', from a repo frozen since
  Jul 2025; at least one (GPQA) is partly contradicted independently.

## 5. Lessons for anyone building on these models

- **Budget host RAM, not just VRAM.** SDAR's load ran out of memory on
  a ~13GB host-RAM runtime while its GPU peak was only 8.87GB — weights
  stage through system RAM. Ollada's recipe now declares an observed
  16GB host-RAM floor for it (observed, not measured as a minimum).
- **The families cannot share one environment.** Transformers pins are
  4.53.3 (Dream), ≥4.57 (SDAR), and saved-under-5.3.0 (DiffCoder).
  Ollada's answer is per-family isolated worker processes; plan for
  isolation from the start.
- **Measure peaks yourself.** Listed specs are estimates, and they can
  be far off in both directions. DiffCoder's 15.15GB measured peak on a
  15.6GB card is this shelf's floor — a number no model card states.
  (The vision side has it worse: a comparable project's changelog
  records a model listed at 10GB measuring 32.1GB.)
- **A fit check needs headroom, not just a peak.** A card also drives
  your displays and desktop. Ollada's fit check reserves 1GB over the
  measured peak before it calls a model comfortable.

## 6. Still open

- **RTX 5060 Ti re-verification** — the pre-launch gate. The live
  question is whether DiffCoder's 15.15GB peak fits a 16GB card that
  is also driving displays, or needs a leaner desktop load profile.
  Speeds, the BF16-vs-fp16 recipe question, and Ollada's fit verdicts
  all get their desktop answers in the same pass.
- DiffCoder `gen_length` sweep (the canvas-exhaustion finding, §2).
- `nemotron:3b` — first planned shelf addition, landing with the
  isolated-worker build its transformers ≥5 requirement forces.


## 7. Addendum (2026-10-23): SDAR on CPU — no GPU at all

Two days after the T4 work, we asked whether any of the shelf could
run on a machine with no GPU at all: a Dell Pro 5 14 (Intel Core
Ultra 7 365, 8 cores, 32GB RAM, Windows 11) — an ordinary managed
work laptop. For SDAR the answer is yes, measured end-to-end through
Ollada's own adapter, bf16, no quantization:

| Model | Train problem (reasoning) | One-word probe | Fibonacci code | Peak host RAM |
|---|---|---|---|---|
| `sdar:1.7b` | 21.2s | 1.9s | 6.3s | 3.9 GB |
| `sdar:4b` | 63.5s | 2.4s | 16.2s | 8.6 GB |

Fixed prompt set, `gen_length` 128, recipe defaults (block 4, 4
denoising steps); all six answers correct. Interactive terminal
sessions followed naturally: short questions answer in seconds,
long-form answers take minutes — and truncated at the 256-token
default canvas, which is why `ollada run` gained a canvas-size
control (`--length`, `/length` in the REPL) and a live canvas view
for SDAR (terminal runs now show the denoising itself: committed
words, current guesses, untouched masks).

- **The step floor is real at both sizes.** Halving denoising steps
  to 2 did not meaningfully save time (the per-block machinery
  dominates) and broke output — the 1.7B's one-word probe never
  commits, and Fibonacci garbles at both sizes. Four steps, SDAR's
  paper protocol, is the working point. Canvas length, not steps,
  is the speed lever.
- **Block diffusion is why this works at all.** SDAR forwards only
  the active block and caches the committed prefix exactly, so a CPU
  answer costs roughly steps-per-block × an autoregressive pass —
  not hundreds of full-canvas passes. In our literature pass
  (2026-10-23) we found no published CPU measurements for any
  block-diffusion model; the only CPU numbers we could find anywhere
  were for LLaDA-8B.
- **Windows took four more SDAR repairs.** The 1.7B's repo ships an
  older modeling revision that imports a transformers symbol removed
  in 4.57 (the adapter aliases it) and lacks the fused-loss file the
  4B's repo also lacks — restored from the authors' GitHub, under a
  different folder than the 4B's. That file imports Triton, which
  has no Windows build (the adapter stubs it; the loss is
  training-only, and its kernels raise if ever called). And the 4B
  modeling's `@torch.compile` decorator must be neutralized during
  load on Triton-less machines, or backend resolution walks into the
  stub. Also: the 4B modeling imports `einops` — invisible on Colab,
  which provides it ambiently; a clean environment has to declare it
  (the recipe now does).
- **Scope, honestly:** one model family, one machine. Dream is
  full-canvas diffusion — every step reprocesses the whole sequence —
  and we expect it to be impractical on CPU; it and DiffCoder are
  untested there and remain GPU models on the shelf. `sdar:1.7b`
  joins the catalog on the strength of this run, its recipe
  labelled CPU-verified; the 5060 Ti re-verification in the status
  header is unchanged and still pending.
