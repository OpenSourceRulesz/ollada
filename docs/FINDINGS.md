# Findings — field notes from putting diffusion LLMs on a shelf

**Status: early.** Everything below comes from Ollada's own real-load runs
through Ollada's own code: a Colab **Tesla T4** (2026-10-21, §1–§6), a
laptop CPU addendum (§7, 2026-10-23), and a desktop re-verification on
an **RTX 5060 Ti 16GB** (§8, 2026-10-25) — the machine this shelf is
actually aimed at. The desktop verification gate that §1–§6 left open
is now closed; the T4 numbers remain as the historic floor, and §8 is
where the target-card numbers live. (Launch/announcement is a separate
decision and is not claimed here.) Where a number is a vendor's, it is
labelled a vendor's. Vendor benchmarks are never presented as our
measurements.

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
run on a machine with no GPU at all: a laptop (Intel Core Ultra 7
365, 8 cores, 32GB RAM, Windows 11) — an ordinary work laptop. For SDAR the answer is yes, measured end-to-end through
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

## 8. Addendum (2026-10-25): desktop GPU verification — RTX 5060 Ti 16GB

The pre-launch gate from §1 is closed. All three GPU models were run
through Ollada's own code on a desktop card (RTX 5060 Ti, 15.9 GB VRAM
by `nvidia-smi`, 31.6 GB host RAM, Windows 11, display attached). The
morning pass ran the fixed prompt set from a main-venv install; the
evening pass ran variant prompts through per-family isolated worker
environments (transformers 4.57.6 for dream/sdar, 5.3.0 for diffcoder;
torch 2.11.0+cu128). Fixed set, verbatim: train "45 km in 30 minutes"
(expect 90 km/h), capital "exactly one word" (expect Paris), fib "only
output code". Evening variants: train "60 km in 45 minutes" (expect
80), capital "one word" (no "exactly"), fib "with a docstring".

| Model | Desktop peak | T4 peak | Fit on the 15.9 GB card |
|---|---|---|---|
| `dream:7b` | 9.13 GB | 9.80 GB | fits (~10.1 GB with headroom) |
| `sdar:4b` | 8.34 GB | 8.87 GB | fits (~9.3 GB with headroom) |
| `diffcoder:8b` | 13.65 GB | 15.15 GB | fits, but tight (~14.7 GB with headroom, ~1.2 GB spare) |

- **Dream is the noisy one.** Morning fixed set, all correct: train
  90 km/h (33.4s), capital exactly "Paris" (30.6s), Fibonacci clean and
  recursive (30.6s). But the wider desktop run set caught failures on
  the same fixed set at the same 128-step default: train wrong twice
  (3 km/h, no minutes-to-hours conversion) and a syntactically broken
  Fibonacci (elif line missing colon and body). Dream samples at
  temperature 0.2, so this reads as sampling noise, not a deterministic
  break — and it matches the T4 sweep (§2), where 128 steps was clean
  but 256 answered the train problem wrong. Conclusion: noisy, not
  deterministically broken — but the variable one of the three, where
  SDAR and DiffCoder were steady on the fixed set. Dream takes ~30s
  per answer at the recipe defaults because it denoises the full
  256-token canvas in 128 steps every time, even for "Paris" — that is
  the default's price, not a bug, and it is faster than the T4's ~38s
  for the same canvas.
- **SDAR is steady.** Fixed set: Fibonacci recursive and correct
  (3.8s). Variant prompts, all correct: 60/45 train → 80 km/h (27.0s),
  fib with docstring correct twice (30.3s, 44.2s), one-word capital
  exactly "Paris" (13.0s after the worker build). Peak 8.34 GB is
  stable across 15 observations. The §6 fit question is answered for
  SDAR: comfortably fits the target card.
- **DiffCoder is steady on the fixed set, wobbly on variants.**
  Fixed set: train 90 km/h with clean working (240.2s first answer
  including first-run warmup, 26.0s on the identical repeat), capital
  exactly "Paris" (0.7s, early stop), Fibonacci iterative and correct
  (12.3s). Variants: the 60/45 train problem came out WRONG (40 km/h
  via 45/60 = 3/2 hours — it is 3/4; Dream and SDAR got 80 on the same
  prompt); the docstring fib came out correct but cut off mid-code
  (the known 256-canvas limit, §2); "one word" without "exactly"
  produced a verbose Paris-plus-explanation instead of the bare word.
  The fixed set stays steady; variants wobble — prompt sensitivity,
  not a broken model.
- **Correction to an earlier stamp.** The first desktop stamp
  (5 observations) put DiffCoder at 9.17 GB and called the T4 tight-fit
  worry resolved. Later runs measured higher: the peak over 13
  observations is 13.65 GB, so the tight-fit story stands, corrected
  (~1.2 GB spare on a display-driving card, not ~5.7). Peak is a
  maximum — the latest max stands. The recipe and the site now carry
  13.65 GB.
- **The API path works on the desktop.** `ollada serve` on
  127.0.0.1:11435; POST /api/chat with `sdar:4b` answered "Paris",
  done=True, eval_count 256, eval_duration 1.917s (early stop on a
  one-word answer, same behavior as the T4's 1.2s precedent).
- **Two install findings from the pass itself.** A fresh `[gpu]`
  install lands transformers 5.x, which breaks Dream's vendored
  modeling with `KeyError: 'default'` (its RoPE config predates the
  5.x rework) — 4.57.6 restores it and keeps SDAR working. And torch
  2.14.1 from plain PyPI was CUDA-blind on this card
  (`torch.cuda.is_available()` False, generation falling back to CPU)
  — 2.11.0+cu128 from the PyTorch CUDA index fixed it. Both are now
  encoded in the per-family worker environments; a bare `[gpu]`
  reinstall would re-break Dream and is not a supported path on this
  card. The worker environments are the 0.3.0 line and are not merged
  to main as of this writing; the numbers above were produced through
  them, and the recipes record the envs that actually ran.

What the desktop changes: DiffCoder's §6 fit question is answered
(tight but fits — close other GPU apps first on a display-driving
card); Dream's speed story stays "about 30 seconds at defaults" with
the noise caveat attached; SDAR's desktop numbers confirm its T4
numbers transfer. What it does not change: the CPU story (§7) stands
as written, and the pending SDPA experiment (laptop, §7's speed
baseline) is still unrun.

**Correction (2026-10-26).** Two lines in this addendum are stale as of this writing: the worker environments were merged to main on 2026-10-25 and shipped in 0.3.0 on 2026-10-26, and the SDPA experiment ran on 2026-10-25 — SDPA did not win on CPU (fib 89% slower), so eager flex stays the default. The desktop numbers above stand.

## 9. Addendum (2026-10-26): Nemotron-3B on CPU — NVIDIA joins the shelf

The shelf's fifth model, and its second CPU-verified entry. The run that
earned it was on a laptop (Intel Core Ultra 7 365, 8 cores, 32 GB RAM,
Windows 11; CPU only, bf16) through ollada's own adapter in an isolated
nemotron worker environment — not the desktop GPU run the probe sheet
originally specified, because the desktop path did not exist on that
machine and the laptop run answered the harder question anyway.

- **The pin is the story.** The first load failed at model-code import:
  NVIDIA's vendored `modeling_ministral.py` imports `check_model_inputs`
  from `transformers.utils.generic`, which transformers 5.3.0 removed
  (the pin borrowed from the diffcoder env). Reading the actual wheels
  settled it: 4.57.6 defines the symbol, 5.3.0 lacks it, 5.20.2 restores
  it as a backward-compat shim. The recipe pins `==5.20.2`. NVIDIA's
  card floor (`transformers>=5.0.0`) is necessary but not sufficient —
  5.3.0 satisfies it and breaks. The remaining load warnings (rope
  FutureWarning, an unrecognized `llama_4_scaling_beta` yarn key, a
  `cache_position` docstring lint, the shim's own deprecation notice)
  are noise; the run works through all of them.
- **Fixed prompt set, all correct, at recipe defaults** (times include
  cold load per CLI run): capital exactly "Paris" (17.9s), train
  90 km/h with clean working, run twice with identical text (62.1s,
  57.3s), Fibonacci iterative and correct (19.0s).
- **Mini code eval, 11/12** (`scripts/mini_code_eval.py`: 12 easy
  HumanEval-style tasks x sdar:1.7b, sdar:4b, nemotron:3b, fresh CLI
  runs). Totals: sdar:1.7b 10/12, sdar:4b 11/12, nemotron:3b 11/12.
  Nemotron's miss was count_vowels, which produced no callable
  function — a format miss, not a wrong answer. Its bright spot is the
  one task both SDARs failed, merge_dicts_sum, which it passed. Speed
  is a wash at the median (nemotron 26.3s, sdar:4b 25.3s per task).
  The shelf decision was contested in the open: the pre-registered
  rule (beat sdar:4b by 3+ tasks or sweep a category) said stay out,
  and the tie said the same. The case that carried it in was not the
  eval but the shelf as a whole: quality parity with the best model,
  proven fallback diversity, CPU depth for the mass-market audience,
  and the shelf's first industry-lab name — NVIDIA recognition is a
  real adoption axis no other shelf model has. The count_vowels miss
  stays on the record, not hidden.
- **What this run corrects.** A secondary source claimed CUDA-capable
  GPU hardware is required for this model. It loads and answers on a
  CPU-only laptop. The claim is contradicted by measurement.
- **What it does not claim.** No GPU verification has been run for
  this model (the desktop pass never happened), no VRAM peak and no
  host-RAM peak were recorded (the CPU run records neither), and no
  tok/s figure is published (times, not token counts, were recorded).
  The recipe's `min_host_ram_gb` of 16 is a prudent floor from the
  7.66 GB weights plus runtime, not a measurement.

- **Speed, measured properly at last.** The CLI runs recorded times but
  no token counts, so no tok/s figure was published for this model.
  The server route supplies both: fib via `/api/chat` on the laptop
  returned eval_count 43, eval_duration_s 5.355 — 8.0 answer tok/s,
  nfe 8. Two things matter here. First, 5.355s is generation time; the
  CLI's 19.0s for the same prompt was mostly load, so repeated use
  (server, REPL) feels much faster than one-shot CLI runs. Second,
  43 is answer tokens, not the 256-token canvas, so 8.0 is answer
  tok/s and is NOT comparable to the site's canvas tok/s figures for
  the other models — it is recorded here, not in the site's Speed
  column, for exactly that reason.
