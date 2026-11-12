# ollada-rs — the ollada Rust front-end (Phase 3)

Read-only shell (`list`, `show`, `recommend`, `ps`, `rm`, plus the fit
check), worker environment management (`workers`, `workers rebuild`),
and the model manager (`run`) speaking the JSON-lines worker
protocol to the real Python workers. No server yet, no model code
moves, nothing the user depends on changes. The Python CLI stays the
reference implementation; this crate must match its output
byte-for-byte — see the parity tests in `src/commands.rs` and
`src/workers.rs` (captured from `python -m ollada.cli` on the
sandbox, including live refusal messages against fake venv pythons)
and the live `sdar:1.7b` runs in the Parity check section below.

## Why this exists

The sketch at `~/workspace/research_notes/ollada-rust-frontend-sketch-20261111.md`:
one binary the user downloads, no Python version, no venvs, no pip, no
torch index, no ensurepip bug. Phase 0 is the smallest first piece: the
commands that never touch a model. Phase 1 (worker setup) follows: the
venv creation, pinned installs, torch-index selection, and ensurepip
recovery from `ollada/workers.py`, orchestrated with `std::process`.
Phase 3 is the model manager from `ollada/runtime/manager.py`: it
spawns the per-family worker subprocesses and speaks the JSON-lines
protocol (`ollada/worker.py`), so the user still never sees the venv.
Python is still what gets installed inside the venv, but the user never
runs pip by hand and never sees the venv. If either phase feels wrong,
each is a small, deletable experiment.

## Layout

- `src/main.rs` — clap CLI, dispatch, exit codes (0 ok, 1 ps-no-daemon,
  2 recipe error — same as Python).
- `src/recipes.rs` — catalog loading/validation, `spec_satisfied`
  (port of `ollada/recipes.py`).
- `src/fit.rs` — fit check: peak trust order (local > verified >
  weights-size estimate), 1 GB headroom, fits/tight/won't-fit/unknown,
  GB10 `[N/A]` handling, nvidia-smi parsing (port of `ollada/fit.py`).
- `src/manager.rs` — the model manager (port of
  `ollada/runtime/manager.py`, worker half only): `WorkerAdapter`
  (spawn, ready handshake, one-JSON-per-line requests, step-event
  forwarding, terminate-then-kill, atexit kill-all) and `ModelManager`
  (one model at a time, fit refusal before eviction, env build before
  eviction, peak recording, serial generation). The future server
  uses this module directly; the CLI's `run` is a thin wrapper.
- `src/measurements.rs` — local measurement records (port of
  `ollada/measurements.py`): read side plus the write side
  (`record_peak`, max-wins, atomic rename). There is no
  `capture_and_record` — that reads the manager's own in-process
  torch, and the Rust manager has no in-process path: a
  worker-reported peak of `None` (CPU, no CUDA) records nothing.
- `src/pyrepr.rs` — Python `repr()` formatting, so `show`'s `generate:`
  / `verified:` lines and `ps`'s dict output match Python exactly.
- `src/cache.rs` — HF cache lookup for `rm` (same layout and size
  accounting as `huggingface_hub.scan_cache_dir`).
- `src/http.rs` — std-only minimal HTTP GET for `ps`.
- `src/commands.rs` — the commands, written to a sink so tests can
  capture them. Now also `workers` / `workers rebuild` and `run`
  (one-shot + REPL loop, same output shape as Python's `_cmd_run`).
- `src/workers.rs` — worker envs (port of `ollada/workers.py`): one
  venv per family under `~/.ollada/workers/<family>/`
  (`$OLLADA_WORKERS_DIR` overrides), pinned installs, the union of the
  family's extras, CPU/CUDA torch-index selection (cu128, cu130 for
  GB10, `$OLLADA_TORCH_INDEX_URL` wins), the ensurepip retry
  (`--without-pip` + get-pip.py bootstrap), the post-build CUDA check,
  the sibling-pin guard, and the rebuild refusal when the env survives
  deletion. Every subprocess call goes through a `Runner` seam, so
  tests observe calls without network or real venvs; the one live
  subprocess test builds a real venv with the real runner.

## Dependencies (current stable, 2026-11-11)

- clap 4.6.7 (derive) — CLI parsing
- serde 1.0.229 (derive), serde_json 1.0.154, serde_yaml 0.9.34 — data
- sysinfo 0.40.1 — host RAM (cross-platform) and killing a timed-out
  nvidia-smi
- libc 0.2 — Unix SIGTERM for worker termination, and the atexit
  handler that kills stragglers (already a transitive dependency via
  sysinfo)

No HTTP client, no regex, no torch: the fit check goes straight to
nvidia-smi, which ships with the NVIDIA driver on Windows and Linux.

## Catalog discovery

`$OLLADA_CATALOG_DIR`, else walk up from the executable looking for
`ollada/catalog` (the repo layout), else `./ollada/catalog` from the
working directory, else `./catalog`.

## Deliberate differences from Python

- **No torch path for GPU detection.** Python tries torch first, then
  nvidia-smi. This crate has no torch, so it goes straight to
  nvidia-smi. On a machine with torch but no nvidia-smi in PATH, Python
  would detect a GPU and this crate would not. In practice nvidia-smi
  ships with the driver wherever a GPU is.
- **System Python discovery.** Python builds venvs with
  `sys.executable` — the interpreter already running ollada. A Rust
  binary has none, so the system Python is found via `$OLLADA_PYTHON`,
  else the first `python3`/`python` on `PATH` (Unix) or
  `python`/`python3`/`py` (Windows). If none is found, venv creation
  fails with a plain error naming the search.
- **get-pip.py download.** Python downloads it with `urllib`
  in-process. This crate has no HTTP client, so it runs the system
  Python itself (`python -c "import urllib.request; ..."`) through the
  same runner seam as every other subprocess call, then runs the script
  with the venv python — same bytes, same endpoint, no new dependency.
- **Probe timeout.** Python's `installed_transformers` /
  `installed_distributions` pass `timeout=60` to `subprocess.run`.
  Here the probes go through `Runner::run_timeout` (60 s); the live
  runner kills the child on timeout (best-effort: if the waiter thread
  already took the child, the kill cannot reach it). Pip installs and
  venv creation have no timeout in either language.
- **`show`'s `measured here` line.** Python crashes with KeyError if a
  record lacks `observations`; this crate prints 1. Real records always
  carry it.
- **The binary is `ollada`.** It IS the `ollada` command when installed
  from the wheel (maturin refuses a wheel with both console scripts
  and a binary, so the binary takes the command name). `--version`
  prints `ollada 0.3.7`, matching Python. `pull` shells out to
  `python -m ollada.cli pull` — downloading weights is the Python
  `huggingface-hub` path.
- **Float formatting.** Rust's `{:?}` matches Python's repr for the
  magnitudes ollada prints (`16.0`, `6.59`, `0.9`); very large/small
  magnitudes could differ (`1e22` vs `1e+22`). Ollada never prints those.
- **Python's `round()` is banker's rounding; Rust's is half-away.**
  The fit check rounds `peak + 1.0` to 2 decimals; no real peak lands
  on an exact half-cent, so this never fires in practice.
- **No in-process path.** Python's `OLLADA_INPROCESS=1` restores
  in-process adapter loading. The Rust manager cannot do that
  (adapters are Python), so a set `OLLADA_INPROCESS` is a clear
  refusal (`error: OLLADA_INPROCESS is set, but in-process loading
  is not supported by the Rust manager ...`, exit 1), never a silent
  fallback.
- **Startup failure kills the worker.** Python's `_start` raises on
  a bad ready line and leaves the spawned process in `_LIVE` until
  process exit. Rust kills it immediately, so a broken spawn never
  lingers.
- **A non-object first line is a clean error.** Python calls
  `.get("event")` on the parsed first line and crashes with
  `AttributeError` if it is a JSON array or scalar. Rust returns a
  `WorkerError` instead.
- **Terminate on Unix, kill on Windows.** Python's
  `proc.terminate()` is SIGTERM on Unix and TerminateProcess on
  Windows. Rust has no portable terminate, so Unix sends SIGTERM
  (via libc) and waits up to 10 s before SIGKILL; Windows calls
  `Child::kill()` (TerminateProcess) and waits. Same observable
  outcome: the worker is gone after unload.
- **No `capture_and_record`.** Python records a CUDA peak from its
  own in-process torch when a chat result carries no peak. The Rust
  manager has no torch; a worker-reported peak of `None` (CPU, no
  CUDA) simply records nothing — which is what Python's capture
  returns on a CPU box too.
- **`run` has no watch / spinner.** Python shows a live diffusion
  canvas on a tty (`_CanvasWatch`) and a `working… Ns` spinner on
  stderr. This prints only the answer on stdout and the
  `({elapsed:.1f}s)` footer on stderr — the same two lines a piped
  Python run produces. The step protocol the watch needs is in the
  manager (`StepHandler`); a renderer is future work. Ctrl-C in the
  REPL kills the process (Python catches it and exits 0); the atexit
  hook still kills the worker.
- **`record_peak`'s `updated` day is UTC.** Python's is local time;
  the two differ only for runs near midnight UTC, and `updated` is
  advisory only.

## Windows

No Unix-only assumptions: no shelling out to Unix tools, no `HOME`-only
paths (`%USERPROFILE%` is honoured), RAM via sysinfo, GPU via
nvidia-smi (same CLI on Windows), venv pythons at
`Scripts/python.exe`, system-Python search includes `py.exe`. The one
`/proc` read (telling a zombie from a live worker in the atexit wait
loop) is `target_os = "linux"`-gated; other Unix falls back to
`kill(pid, 0)`, Windows kills by pid via sysinfo.
Cross-compiling and testing on Windows is still unverified — that is
the known open risk from the sketch, not a claim made here. The
Windows target does compile and pass clippy
(`cargo check --target x86_64-pc-windows-gnu`), but no Windows binary
has been run.

## Verify

```sh
cargo fmt --check
cargo check
cargo clippy --tests -- -D warnings
cargo test
cargo check --target x86_64-pc-windows-gnu
cargo clippy --target x86_64-pc-windows-gnu --tests -- -D warnings
```

## Parity check (against the Python CLI, same machine)

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build
for cmd in "list" "recommend" "show diffcoder:8b4bit" "show sdar:1.7b" "ps" "rm diffcoder:8b4bit"; do
  diff <(python -m ollada.cli $cmd) <(./target/debug/ollada $cmd) && echo "OK: $cmd"
done
for cmd in "workers" "workers list" "workers rebuild nope:1b"; do
  diff <(OLLADA_WORKERS_DIR=/tmp/empty-workers python -m ollada.cli $cmd 2>&1; echo "exit=$?") \
       <(OLLADA_WORKERS_DIR=/tmp/empty-workers ./target/debug/ollada $cmd 2>&1; echo "exit=$?") && echo "OK: $cmd"
done
```
(Run from the repo root with the Python venv's `ollada` on PATH. A full
torch install is never run live in the sandbox: the build path is
verified with a fake runner in `src/workers.rs` plus one real
`python -m venv` via the real runner. `rm` needs `huggingface-hub` in
the Python environment to compare; the system python3 here lacks it,
so that one diff is environmental, not a code difference.)

Live `run` parity (2026-11-11, sandbox, CPU torch, `sdar:1.7b`):

- Fit refusal, no `--force`: stderr byte-identical, exit 2 both
  (`fit check: sdar:1.7b declares a 8 GB host-RAM floor ... this
  machine has 7.7 GB ... Override with --force ...`).
- `--length 0`: `length must be a positive number of tokens`, exit 2 both.
- `run sdar:1.7b "What is the capital of France?" --force`: stdout
  byte-identical across three runs (Python 362.8s, Rust 21.5s, Python
  again 22.1s — all `The capital of France is Paris.`). The first
  Python run's 363 s is sandbox CPU-throttling variance, not a
  Python/Rust difference: the repeat Python run took 22.1 s. No Rust
  speedup is claimed. Stderr footer is the same `({elapsed:.1f}s)`
  shape in all three; the worker's `resource_tracker` warning appears
  in all three (stderr is inherited, like Python). No worker process
  is left behind in any run; no measurement record is written in any
  run (CPU, peak `None`).
- REPL, same piped input (`What is the capital of France?`,
  `/length 32`, `Say hello in one short sentence.`): stdout
  byte-identical Python vs Rust (banner, `>>> ` prompts, both
  answers, `canvas length: 32 tokens`, trailing `>>> ` on EOF);
  stderr differs only in the timing values.
- `OLLADA_INPROCESS=1 ... run sdar:1.7b "hi" --force`: Rust refuses
  with the in-process message above, exit 1. (Python would take the
  in-process path; there is no Python equivalent to compare.)
