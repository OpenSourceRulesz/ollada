# Diffusion Shelf — Plan
### Working name: **Ollada** (see §9 — mirrors Ollama's naming move: Ollama ← LLaMA, Ollada ← LLaDA). An "Ollama for diffusion models": a curated library + local service where people know to get open-source diffusion language models (dLLMs).
**Status:** Plan v1.2 — 2026-10-21 — **BUILD STARTED** at Dallas's go (3-model shelf with the Nemotron swap, §5). **Name CLAIMED on GitHub + PyPI**: github.com/OpenSourceRulesz/ollada created + pushed; Dallas registered the PyPI pending trusted publisher; tag v0.0.1 pushed — note the tag push never triggered the workflow (0 runs; cause undetermined), so the publish ran via manual `workflow_dispatch` from main after the version-check step was made tag-only; run 39617965181 succeeded; `pip install ollada` (0.0.1) verified from a fresh venv AND via `pip download --no-cache-dir`; PyPI JSON API returns 200 (its initial 404 was CDN negative-caching from the pre-claim checks). Runtime build still awaits the remaining plan decisions.
**Cost rule:** $0 project (same as FFTP / PrivateContext). No domain, no telemetry, no paid services. Gemini free tier only if ever needed. Nothing public without per-item approval (draft → review → agree → post).

---

## 1. The insight this is built on
- People don't get AR models from a raw engine, they get them from a *shelf*: Ollama's library. Clean short names, a model is either on the shelf (tested, quantized, templated, safe) or it isn't.
- Diffusion has no shelf. As of Oct 2026: no free browser demo that reliably works (except one community LLaDA Space), no canonical names, every Hugging Face repo has a different `generate()` signature, most need `trust_remote_code=True`, and defaults (steps / block size / threshold) that make or break quality are left to the user to guess.
- HF is the warehouse (storage). This project is the store (curation + working recipes + a local API tools can point at).

## 2. Product definition (MVP)
One command installs it, one command runs a diffusion model, models have short names, and a local HTTP API serves them.

```
pip install <name>
<name> pull diffcoder:8b
<name> run diffcoder:8b
<name> serve            # http://localhost:11435 (not 11434 — avoids the known Ollama/VS Code clash on Dallas's Dell)
```

**In scope for v0.1:**
- CLI: `pull / run / list / ps / rm / serve / show`
- Daemon: FastAPI service, one GPU model loaded at a time, keep-alive + auto-unload, serial queue
- APIs: Ollama-shaped `POST /api/chat`, `POST /api/generate`, plus OpenAI-shaped `POST /v1/chat/completions` (final text only in v0.1)
- A model *Recipe* format (the DiffFile — Ollama's Modelfile equivalent, see §4)
- A curated catalog of **3 models** (see §5), each with a *verified* recipe
- A one-click Colab notebook demo — this is the "try it in a browser" gap that started the whole thread
- Static catalog page (GitHub Pages, $0) listing the shelf: task, size, VRAM, verified speed, license

**Explicitly out of scope for v0.1:**
- C++ / GGUF engine (that's Tier 3 — team + a year; Python + PyTorch + bitsandbytes is the MVP runtime)
- CPU-first support (diffusion on CPU is minutes-per-answer; GPU-first, honest about it. CPU smoke-test only for the 1.7B)
- Canvas-snapshot streaming (v0.2 candidate — the "watch it develop" UI)
- Training / fine-tuning, multi-GPU, batching/concurrency beyond a serial queue
- Any model whose family adapter isn't hand-written and tested by us (no arbitrary repo execution)

## 3. Architecture
```
catalog/            # the shelf: one YAML per model on the shelf (curated, reviewed)
recipes/            # verified generation defaults per model (steps, block, threshold, quant, template)
adapters/           # one hand-written adapter per model FAMILY (load / generate / chat-template)
  dream.py          # reuse the proven npc-engine harness (8-bit load, BitsAndBytesConfig, transformers pin)
  sdar.py           # JetLM SDAR family (custom_code — re-implement its loop, don't execute theirs blindly)
  diffcoder.py      # Stable-DiffCoder family
runtime/
  server.py         # FastAPI daemon, model manager (load/unload/VRAM guard), queue
  cli.py            # Typer CLI
tests/
  test_*.py         # unit tests (mocked) — necessary, NOT sufficient (see §7)
  verify/           # REAL-load verification scripts per model (the gate for "on the shelf")
notebooks/
  colab_demo.ipynb  # fresh-kernel-safe, version-stamped, prints version + "ready on cuda" acceptance line
docs/
  LIBRARY.md        # human-readable shelf, generated from catalog/
```

**Isolation rule (added v1.2, forced by a real conflict):** families have incompatible dependency pins — Dream needs `transformers==4.53.3`, Nemotron needs `transformers>=5.0`. So each family runs in an **isolated worker subprocess with its own pinned environment**; the daemon talks to one worker at a time over a local pipe. This also buys crash isolation (a model segfault can't kill the daemon) and clean VRAM unload (kill the worker). Phase 1 starts in-process; with the SDAR decision (2026-10-21) all three v0.1 families can share the 4.53.3-era environment, so isolation is deferred until the Nemotron addition lands post-v0.1 — it remains required before Nemotron ships, not before v0.1.

Design rules carried over from npc-engine / PrivateContext (learned the hard way):
- Never pass raw quantization kwargs into a model's `__init__` (the `load_in_8bit` crash) — always `BitsAndBytesConfig`.
- Pin transformers *per adapter family* if needed, and say so in the recipe — Dream needed `transformers==4.53.3` (ROPE `'default'` KeyError on 5.x).
- Mocks don't cover real load paths. A model is not "supported" until a real-load verification has passed on real GPU hardware (see §7).
- PowerShell-first for all Windows-facing instructions; one command per fenced block, full args on one line.

## 4. The Recipe (DiffFile) — the core standard we are proposing
```yaml
name: diffcoder:8b
family: diffcoder
source: ByteDance-Seed/Stable-DiffCoder-8B-Instruct
license: MIT
task: coding
size_gb_bf16: 16
quant_default: int8        # ~8-9 GB — the safe default on 16-32 GB machines
generate:
  steps: 128               # vendor example uses 512; our verified default is a measured quality/speed trade
  gen_length: 256
  block_length: 32
  remasking: low_confidence
  temperature: 0.0
verified:
  hardware: T4-16GB (8-bit)
  tokens_per_sec: null     # filled in ONLY by a real verification run — never a vendor number, never a mockup
  date: null
```
Every field a normal user should never have to think about lives here, tested, not in the README as homework.

## 5. The initial shelf (3 models for v0.1, expand only after the gate passes)
| Short name | Source | Job | Why it's on first |
|---|---|---|---|
| `dream:7b` | Dream-org/Dream-v0-Instruct-7B | general chat | We already have a proven, debugged harness for it (npc-engine). Lowest-risk first adapter. |
| `sdar:4b` | JetLM/SDAR-4B-Chat | general | Strongest per-size diffusion generalist in its paper (MMLU 74.9 / GSM8K 89.9 / Math500 72.8, all diffusion mode, greedy), Qwen3-4B converted, Apache-2.0. Setup note: needs only `transformers>=4.52.4` — the shared Dream pin satisfies it, so v0.1 stays a single environment. Its adapter vendors SDAR's reference block-diffusion loop (Apache-2.0, credited) because SDAR ships no `model.generate()`. **Restored to the shelf by Dallas's decision 2026-10-21** (see decision record below). |
| `diffcoder:8b` | ByteDance-Seed/Stable-DiffCoder-8B-Instruct | coding | Best open diffusion coder (HumanEval 86.6 / MBPP 85.7 / BigCodeBench-Full 54.8), MIT. This is the model Dallas asked for for his own coding use. |

**First planned addition after v0.1:** `nemotron:3b` (NVIDIA) — adapter already written and kept in the tree; it lands together with the isolated-worker build its `transformers>=5` requirement forces. **v0.2 shelf candidates (not promises):** `sdar:1.7b`, `sdar:8b`, `dream-coder:7b`, `fast:1.5b` (Fast-dLLM v2), `llada:8b`.

### Decision record — general slot: nemotron:3b vs sdar:4b (reopened by Dallas 2026-10-21, analysis same day)
Diffusion-mode-to-diffusion-mode, from each lab's own paper (different eval setups — indicative, not controlled):
| Metric | SDAR-4B (native diffusion) | Nemotron-3B (diffusion mode only) |
|---|---|---|
| MMLU | **74.9** | 72.1 |
| GSM8K | **89.9** | 88.4 |
| Math500 | 72.8 | **74.8** |
| HumanEval | 72.8 | **74.4** |
| MBPP | 65.4 | **73.5** |
| IFEval | 56.6 | **68.9** |
| GPQA | 33.0 | 33.8 |
Notes: the Nemotron numbers Muse originally quoted (MMLU 72.1 / HumanEval 79.3 / avg 55.8) are its **quad self-speculation** mode, not diffusion (avg 52.9, TPF 1.91); its diffusion mode still beats its own AR mode on MMLU/GSM8K/MBPP in NVIDIA's table. SDAR's numbers are at its fastest config (block 4, 4 steps). Other axes: size ~8.8GB vs ~6.2GB weights; context 32K vs 256K (config); license Apache-2.0 vs NVIDIA Nemotron license; downloads 897 vs ~67k.
**Setup ease (added 2026-10-21 at Dallas's question — it was in the analysis as engineering cost but NOT ranked as its own axis):** (1) Manual install today: Nemotron slightly easier — one requirement (`transformers>=5.0.0`) + copy-paste snippet on its card (though its AR example has a variable bug); SDAR's docs lead with git-clone + `generate.py`, then a separate engine (JetEngine/LMDeploy) for speed. (2) Inside ollada: **SDAR clearly easier** — needs `transformers>=4.52.4`, already satisfied by Dream's 4.53.3 pin, same environment; Nemotron is the sole reason the isolated-worker build lands in v0.1. (3) For the ollada end user: tie by design (both are `ollada run`), except Nemotron's smaller download / 8GB-card fit. Weighting setup ease narrows Muse's lean to **Nemotron ~60/40**; if fastest-time-to-verified-shelf were the top priority, SDAR would win on this axis. **SDAR setup findings (2026-10-21, from Dallas's first verification run + local reproduction of every failure):** the "SDAR is setup-easy" premise did not survive contact. Its HF repo (1) hard-requires `flash_attn` in its RMSNorm forward — no PyPI wheels, no Windows build, never mentioned for the built-in path; (2) is missing `fused_linear_diffusion_cross_entropy.py`, a file its own modeling imports (it exists only in the authors' GitHub training folder; the imported class is a dead import at inference); (3) claims `transformers>=4.52.4` but its modeling imports `TransformersKwargs`, which exists only from **4.57** — so it does NOT share Dream's verified 4.53.3 env; (4) its attention accepts only flex-attention BlockMasks, and no shipped code path converts tensor masks (its own generate.py cannot run as shipped). The ollada adapter repairs all four at load (torch RMSNorm fallback — bit-exact vs manual norm in local tests; authors' file restored from their GitHub into a link overlay; BlockMask conversion in the vendored loop; compiled flex wrapper swapped for the eager call). Verified locally as far as a CPU rig can: both user-visible errors reproduced exactly, import chain + tiny-config instantiation pass on 4.57, loop proven against a contract fake. GPU flex execution is what the rerun tests. **Lesson recorded: "setup ease" claims get verified against the real artifact before they steer decisions — same standard as benchmarks.**
**DECIDED by Dallas 2026-10-21: "flip to sdar for now we can always add later."** SDAR takes the v0.1 general slot (setup ease in our stack was the deciding axis); Nemotron becomes the first planned post-v0.1 addition, landing with the isolated-worker build. The T4 head-to-head is moot for v0.1; the verification gate still applies to SDAR exactly as to every shelf model.

## 6. Phases & rough timeline (solo + Muse, part-time)
- **Phase 0 — Spec freeze (1-2 days, no code beyond skeleton):** pick the public name (see §9), freeze Recipe format + API shapes, repo skeleton, this plan approved. *Gate: Dallas says go.*
- **Phase 1 — One model end-to-end (week 1):** daemon + CLI + Dream adapter, `pull/run/serve` working, real-load verified on Colab T4. Colab demo notebook v1. *Gate: a stranger could run the notebook fresh-kernel and get a good line out.*
- **Phase 2 — The shelf becomes real (week 2):** SDAR + DiffCoder adapters, steps-sweep for defaults (32/64/128/256, measured — quality vs tok/s), verification runs for all 3, catalog page generated. *Gate: all 3 pass §7 on T4; DiffCoder also passes a small real coding-prompt set (not just HumanEval numbers from the paper).* **GATE PASSED 2026-10-21: dream:7b, sdar:4b, and diffcoder:8b all verified on T4 through ollada's own code (numbers + quality notes in their recipes); DiffCoder's run included the 4-prompt coding set. The steps sweep completed the same day: diffcoder default 128 -> 256 (64 collapsed into repetition degeneration; 256's prose is clean where 128's wobbles), dream default 32 -> 128 (32/64 both broke Fibonacci code; 256 answered a math prompt wrong once and is half the speed — more steps was not better). Still owed from this phase: the catalog page.**
- **Phase 3 — v0.1.0 release (week 3):** docs (exhaustive, zero-assumed-knowledge install walkthrough), PyPI publish via trusted publishing, GitHub public. *Each public step follows draft → review → agree → post; one-time tokens never stored.*
- **Phase 4 — Post-launch, only if people use it:** canvas streaming, more models, Continue / OpenWebUI integration snippets, a "verified on 4090" tier.

## 7. Verification gate — what "on the shelf" means
A model earns its short name only when ALL of these are true, from a *real* run we did:
1. Fresh-environment load (fresh Colab kernel / fresh venv) with the pinned deps in its recipe — version stamp printed.
2. `ready on cuda` equivalent acceptance line + actual GPU generation (not CPU fallback).
3. Fixed prompt set (chat, code, infill, short + long) produces sane output at the recipe's default steps.
4. Measured: load time, peak VRAM, tokens/sec, at the default quant — published in the catalog, labelled with the hardware.
5. Known-failure notes written down (context limits, prompt formats that break it, steps below which it falls apart).
If a model fails, it stays off the shelf and the failure is documented. No vendor benchmark is ever presented as our measurement.

## 8. Risks & kill criteria
- **Transformers fragility** (biggest known risk — it bit npc-engine twice): mitigate with per-family pins + the §7 gate on every release. If a family needs a *fork* to stay working, that family gets cut, not the project.
- **Adapter sprawl:** if family #4 requires a materially different runtime (e.g. SGLang-only), stop and re-decide scope instead of growing a second engine quietly.
- **Expectation trap:** people will try it on CPU laptops (like the Dell) and conclude diffusion is slow. Mitigate by positioning GPU-first everywhere, and giving the 1.7B an honest "CPU: try-it speed" label later.
- **Trademark:** do NOT name it anything containing "Ollama" (their mark) or "Llama" (Meta's). Working title "Diffusion Shelf" is descriptive only.
- **Kill criterion:** if, after Phase 2, none of the 3 models gives a *good* experience on a free T4, the honest move is to stop at a research repo + write-up, not ship a shelf we wouldn't use ourselves.

## 9. Decisions needed from Dallas (one round, then Phase 0 starts)
1. **Name** — Dallas's direction (2026-10-21): do Ollama's naming move again. Ollama launched around **LLaMA** (Meta's first big open AR model); the first big open *text-diffusion* model is **LLaDA** (Feb 2025, Renmin / Ant Group — see plan notes). So: **Ollada** (O + Llada). Availability checked 2026-10-21 (404 = free at check time, controls `fftp`/`ollama` correctly returned 200; availability is not a trademark clearance): `ollada` is free on **both PyPI and Hugging Face**. Fallbacks also free on PyPI: `allada`, `lladama`, `dllada`, `diffshelf`, `diffserve`, `dllm`, `maskserve`. Note: LLaDA itself is Renmin/Ant's model name — Ollada is a distinct coinage in the same spirit as Ollama vs Llama, but a real trademark sanity check is a Phase 0 task, not an assumption.
2. **First shelf** — approve the 3-model MVP (dream:7b / sdar:4b / diffcoder:8b), or swap one.
3. **First hardware target** — RESOLVED 2026-10-21: Dallas's personal gaming desktop, an **RTX 5060 Ti 16GB** (variant confirmed by Dallas; Blackwell, 4,608 CUDA cores, 16GB GDDR7 — runs the full 8B shelf in 8-bit comfortably). This is also the archetypal target-user machine: Ollama's audience is people with consumer gaming GPUs on Windows. **Sequencing:** development + first verification happen on **Colab T4 now** (Dallas can't use the desktop at the moment). This does not change the design: the T4 is also 16GB, so every VRAM fit / quantization decision transfers 1:1. What does NOT transfer is speed — all published numbers stay labelled with the hardware they were measured on, T4 numbers will read as pessimistic vs the 5060 Ti, and a re-verification pass on the desktop is a required pre-launch gate. One dtype nuance: the T4 (Turing) has no native bf16 acceleration, so T4 recipes default to fp16 / 8-bit; the 5060 Ti (Blackwell) can use bf16 recipes. The work Dell (32GB, Intel integrated graphics) is explicitly NOT a target — it's a CPU-only smoke-test machine at best.
4. **Public posture** — local-first like npc-engine started (build, verify, then a separate go for public), recommended. No public step is bundled into the build go.

## 10. What success looks like
A developer who has never heard of SDAR can, in under 10 minutes on a free Colab GPU: install it, `run diffcoder:8b`, get good code out, and `serve` it for their editor — and when a friend asks "where do you get diffusion models?", the answer is this project's name.
