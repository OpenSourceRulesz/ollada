//! Worker environments — one venv per model family, pinned from its recipe.
//! A port of `ollada/workers.py`.
//!
//! Why a venv per family instead of one shared env: the families' pins are
//! incompatible (dream needs transformers 4.x, diffcoder needs 5.3.0, sdar
//! needs >=4.57,<5), so no single install satisfies the shelf. A worker venv
//! holds exactly one family's stack. The install set is the UNION of the
//! family's recipes' extras (the sdar family proved why — 1.7b's modeling
//! never imports einops, 4b's does). A pin mismatch or a missing package in
//! an existing env is a refusal, not a silent rebuild: rebuilding
//! re-downloads gigabytes, so it happens only on an explicit
//! `ollada workers rebuild <model>`.
//!
//! Layout: `~/.ollada/workers/<family>/` (`OLLADA_WORKERS_DIR` overrides,
//! same pattern as `OLLADA_MEASUREMENTS_DIR`).
//!
//! Deliberate differences from Python, all documented:
//! - **System Python discovery.** Python uses `sys.executable` — the
//!   interpreter already running ollada. A Rust binary has none, so the
//!   system Python is found via `$OLLADA_PYTHON`, else the first
//!   `python3`/`python` on `PATH` (Unix) or `python`/`python3`/`py` on
//!   Windows. If none is found, venv creation fails with a plain error
//!   naming the search instead of Python's always-present interpreter.
//! - **get-pip.py download.** Python downloads it with `urllib` in-process.
//!   This crate has no HTTP client, so it runs the system Python itself
//!   (`python -c "import urllib.request; ..."`) through the same runner
//!   seam as every other subprocess call, then runs the script with the
//!   venv python — same bytes, same endpoints, no new dependency.
//! - **Probe timeout.** Python's `installed_transformers` /
//!   `installed_distributions` pass `timeout=60` to `subprocess.run`.
//!   The probes here run through `Runner::run_timeout` (60 s); the
//!   live runner kills the child on timeout (best-effort — see
//!   `RealRunner::run_timeout`). Pip installs and venv creation have
//!   no timeout in either language.
//! - **Hardware detection.** `ensure_env` calls `fit::detect_hardware()`,
//!   which goes straight to nvidia-smi with no torch path (see `fit.rs`).
//! - **Cleanup on failure.** Exactly like Python: only venv-creation
//!   failures remove the half-built dir. A pip failure, a no-CUDA torch,
//!   or an incompatible sibling pin leaves the env in place for
//!   inspection and an explicit rebuild.

use crate::fit::{detect_hardware, Hardware};
use crate::measurements::home_dir;
use crate::recipes::{spec_satisfied, Recipe};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

pub const CPU_TORCH_INDEX: &str = "https://download.pytorch.org/whl/cpu";
/// PyPI's torch is CPU-only — a GPU machine that installs torch from
/// PyPI silently lands CPU torch. CUDA wheels live on PyTorch's index.
pub const CUDA_TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
pub const GB10_TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu130";
const GET_PIP_URL: &str = "https://bootstrap.pypa.io/get-pip.py";

/// Mirrors `ollada.workers.WorkerEnvError`.
#[derive(Debug)]
pub struct WorkerEnvError(pub String);

impl fmt::Display for WorkerEnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for WorkerEnvError {}

/// One subprocess invocation's result.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[cfg(test)]
impl RunOutput {
    pub fn new(code: i32, stdout: &str, stderr: &str) -> Self {
        RunOutput {
            code,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
        }
    }
}

/// A subprocess runner: the seam every subprocess call in this module
/// goes through, so tests can observe calls without network or real venvs
/// (mirrors Python's `_run` parameter on `ensure_env` et al.).
pub trait Runner: Send + Sync {
    /// Run `argv` to completion, capturing stdout/stderr. `Err` is a
    /// spawn failure only; a non-zero exit is a normal `Ok` output.
    fn run(&self, argv: &[String]) -> Result<RunOutput, String>;

    /// Run `argv` with a timeout. The default is `run` (no timeout);
    /// `RealRunner` kills the child on timeout. The metadata probes
    /// use this with 60 s, mirroring Python's `timeout=60`.
    fn run_timeout(&self, argv: &[String], _timeout: Duration) -> Result<RunOutput, String> {
        self.run(argv)
    }
}

/// The live runner: `std::process::Command`, no shell, no timeout
/// (pip installs are long; the metadata probes add their own 60 s
/// timeout in `run_probe`).
pub struct RealRunner;

impl Runner for RealRunner {
    fn run(&self, argv: &[String]) -> Result<RunOutput, String> {
        if argv.is_empty() {
            return Err("empty command".to_string());
        }
        let output = Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|e| e.to_string())?;
        Ok(output_to_run_output(output))
    }

    fn run_timeout(&self, argv: &[String], timeout: Duration) -> Result<RunOutput, String> {
        if argv.is_empty() {
            return Err("empty command".to_string());
        }
        let child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        // Shared with the waiter thread so a timeout here can kill
        // the child; the waiter takes it out (leaving None) once it
        // starts waiting, in which case the kill is best-effort only.
        let shared: std::sync::Arc<std::sync::Mutex<Option<std::process::Child>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Some(child)));
        let waiter_child = std::sync::Arc::clone(&shared);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let taken = waiter_child.lock().map(|mut g| g.take()).unwrap_or(None);
            let result = match taken {
                Some(c) => c.wait_with_output().map_err(|e| e.to_string()),
                None => return, // killed from the timeout path; nothing to wait on
            };
            let _ = tx.send(result);
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(output)) => Ok(output_to_run_output(output)),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                if let Ok(mut guard) = shared.lock() {
                    if let Some(mut c) = guard.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                }
                Err(format!("timed out after {}s", timeout.as_secs()))
            }
        }
    }
}

fn output_to_run_output(output: std::process::Output) -> RunOutput {
    RunOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Run a probe through the runner with a timeout. Returns `None` on
/// spawn failure or timeout (the caller treats that as "absent/broken",
/// like Python's `except Exception: return None`).
fn run_probe(runner: &dyn Runner, argv: &[String], timeout: Duration) -> Option<RunOutput> {
    runner.run_timeout(argv, timeout).ok()
}

// -- paths -------------------------------------------------------------------

/// Mirrors `ollada.workers.workers_dir`.
pub fn workers_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLADA_WORKERS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    home_dir().join(".ollada").join("workers")
}

pub fn venv_path_in(dir: &Path, family: &str) -> PathBuf {
    dir.join(family)
}

pub fn venv_python_in(dir: &Path, family: &str) -> PathBuf {
    let root = venv_path_in(dir, family);
    if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    }
}

#[allow(dead_code)] // ported API surface; the CLI passes an explicit dir
pub fn venv_path(family: &str) -> PathBuf {
    venv_path_in(&workers_dir(), family)
}

#[allow(dead_code)] // ported API surface; the CLI passes an explicit dir
pub fn venv_python(family: &str) -> PathBuf {
    venv_python_in(&workers_dir(), family)
}

/// Select a CUDA torch wheel that supports the detected GPU.
/// Mirrors `ollada.workers.cuda_torch_index`.
pub fn cuda_torch_index(hardware: &Hardware) -> String {
    let override_url = std::env::var("OLLADA_TORCH_INDEX_URL").ok();
    cuda_torch_index_with_override(hardware, override_url.as_deref())
}

pub fn cuda_torch_index_with_override(hardware: &Hardware, override_url: Option<&str>) -> String {
    if let Some(url) = override_url {
        if !url.is_empty() {
            return url.to_string();
        }
    }
    // GB10 (SM 12.1) needs CUDA 13.0 wheels on Linux aarch64.
    if hardware
        .gpu_name
        .as_deref()
        .map(|n| n.to_uppercase().contains("GB10"))
        .unwrap_or(false)
    {
        return GB10_TORCH_INDEX.to_string();
    }
    CUDA_TORCH_INDEX.to_string()
}

// -- recipe helpers ----------------------------------------------------------

fn env_transformers(recipe: &Recipe) -> String {
    recipe
        .env
        .get("transformers")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn env_extras(recipe: &Recipe) -> Vec<String> {
    match recipe.env.get("extras").and_then(|v| v.as_sequence()) {
        Some(seq) => seq
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => Vec::new(),
    }
}

fn family_recipes<'a>(family: &str, catalog: &'a [Recipe]) -> Vec<&'a Recipe> {
    catalog.iter().filter(|r| r.family == family).collect()
}

/// Union of extras across the family's recipes, in first-seen order.
fn family_extras(family: &str, catalog: &[Recipe]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for recipe in family_recipes(family, catalog) {
        for extra in env_extras(recipe) {
            if !out.contains(&extra) {
                out.push(extra);
            }
        }
    }
    out
}

/// The dedupe key Python uses: the package name before any `[`, `=`,
/// `>`, or `<`.
fn pkg_key(pkg: &str) -> String {
    pkg.split('[')
        .next()
        .unwrap_or(pkg)
        .split('=')
        .next()
        .unwrap_or("")
        .split('>')
        .next()
        .unwrap_or("")
        .split('<')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Install set for a family's worker env, order-deduped.
/// Mirrors `ollada.workers._pip_packages`.
pub fn pip_packages(recipe: &Recipe, catalog: &[Recipe]) -> Vec<String> {
    let spec = env_transformers(recipe);
    let mut pkgs = vec![
        "pyyaml".to_string(),
        "torch".to_string(),
        format!("transformers{spec}"),
        "accelerate".to_string(),
    ];
    pkgs.extend(family_extras(&recipe.family, catalog));
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for pkg in pkgs {
        let key = pkg_key(&pkg);
        if seen.insert(key) {
            out.push(pkg);
        }
    }
    out
}

fn norm_dist_name(name: &str) -> String {
    name.to_lowercase().replace('-', "_")
}

// -- installed-package probes ------------------------------------------------

/// Installed transformers version in a venv, or None if absent/broken.
/// Uses importlib.metadata, not an import: importing transformers is
/// slow and can fail for reasons unrelated to the version question.
pub fn installed_transformers(python: &Path, runner: &dyn Runner) -> Option<String> {
    if !python.exists() {
        return None;
    }
    let argv = vec![
        python.to_string_lossy().into_owned(),
        "-c".to_string(),
        "import importlib.metadata as m; print(m.version('transformers'))".to_string(),
    ];
    let out = run_probe(runner, &argv, Duration::from_secs(60))?;
    if out.code != 0 {
        return None;
    }
    let version = out.stdout.trim();
    if version.is_empty() {
        None
    } else {
        Some(version.to_string())
    }
}

/// Installed distribution names in a venv, normalized, or None.
/// One importlib.metadata scan, no imports of the packages themselves.
pub fn installed_distributions(python: &Path, runner: &dyn Runner) -> Option<HashSet<String>> {
    if !python.exists() {
        return None;
    }
    let argv = vec![
        python.to_string_lossy().into_owned(),
        "-c".to_string(),
        "import importlib.metadata as m; print('\\n'.join(d.metadata['Name'] for d in m.distributions()))"
            .to_string(),
    ];
    let out = run_probe(runner, &argv, Duration::from_secs(60))?;
    if out.code != 0 {
        return None;
    }
    Some(
        out.stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| norm_dist_name(line.trim()))
            .collect(),
    )
}

/// True only if the worker env's torch affirmatively reports CUDA.
fn worker_torch_has_cuda(python: &Path, runner: &dyn Runner) -> bool {
    let argv = vec![
        python.to_string_lossy().into_owned(),
        "-c".to_string(),
        "import torch; print(torch.cuda.is_available())".to_string(),
    ];
    match runner.run(&argv) {
        Ok(out) => out.code == 0 && out.stdout.trim() == "True",
        Err(_) => false,
    }
}

// -- subprocess helpers ------------------------------------------------------

/// Last 5 lines of stderr (else stdout), joined with " | ".
/// Mirrors Python's `(proc.stderr or proc.stdout or "").strip().splitlines()[-5:]`.
fn tail(out: &RunOutput) -> String {
    let text = if !out.stderr.is_empty() {
        &out.stderr
    } else {
        &out.stdout
    };
    let lines: Vec<&str> = text.trim().split('\n').collect();
    let n = lines.len();
    let start = n.saturating_sub(5);
    lines[start..].join(" | ")
}

fn run_checked(runner: &dyn Runner, argv: &[String], what: &str) -> Result<(), WorkerEnvError> {
    let proc = runner
        .run(argv)
        .map_err(|exc| WorkerEnvError(format!("could not {what}: {exc}")))?;
    if proc.code != 0 {
        return Err(WorkerEnvError(format!(
            "could not {what} (exit {}): {}",
            proc.code,
            tail(&proc)
        )));
    }
    Ok(())
}

/// The system Python to build venvs with. Python uses `sys.executable`;
/// a Rust binary has none, so: `$OLLADA_PYTHON`, else the first
/// `python3`/`python` on `PATH` (Unix) or `python`/`python3`/`py` on
/// Windows.
pub fn system_python() -> Result<PathBuf, WorkerEnvError> {
    if let Ok(p) = std::env::var("OLLADA_PYTHON") {
        if !p.is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    let candidates: &[&str] = if cfg!(windows) {
        &["python.exe", "python3.exe", "py.exe"]
    } else {
        &["python3", "python"]
    };
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            for cand in candidates {
                let p = dir.join(cand);
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    Err(WorkerEnvError(format!(
        "no system Python found on PATH (tried {}) — install Python or set OLLADA_PYTHON",
        candidates.join(", ")
    )))
}

fn remove_dir_all(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
}

/// Install pip into a --without-pip venv via get-pip.py.
/// Standard recovery when a venv's own ensurepip is broken (Colab,
/// 2026-11-06). Downloads get-pip.py from bootstrap.pypa.io — via the
/// system Python's urllib, through the runner seam — and runs it with
/// the venv python. Raises WorkerEnvError if the download or the run fails.
fn bootstrap_pip(
    python: &Path,
    sys_python: &Path,
    runner: &dyn Runner,
) -> Result<(), WorkerEnvError> {
    let tmp = std::env::temp_dir().join(format!(
        "ollada-get-pip-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp).map_err(|exc| {
        WorkerEnvError(format!(
            "could not bootstrap pip into the worker env: {exc}"
        ))
    })?;
    let script = tmp.join("get-pip.py");
    let result = (|| -> Result<(), WorkerEnvError> {
        let dl_argv = vec![
            sys_python.to_string_lossy().into_owned(),
            "-c".to_string(),
            format!(
                "import urllib.request; urllib.request.urlretrieve({GET_PIP_URL:?}, {:?})",
                script.to_string_lossy()
            ),
        ];
        match runner.run(&dl_argv) {
            Err(exc) => {
                return Err(WorkerEnvError(format!(
                    "could not bootstrap pip into the worker env: {exc}"
                )));
            }
            Ok(out) if out.code != 0 => {
                return Err(WorkerEnvError(format!(
                    "could not bootstrap pip into the worker env: download failed (exit {}): {}",
                    out.code,
                    tail(&out)
                )));
            }
            _ => {}
        }
        run_checked(
            runner,
            &[
                python.to_string_lossy().into_owned(),
                script.to_string_lossy().into_owned(),
            ],
            "bootstrap pip into the worker env",
        )
    })();
    remove_dir_all(&tmp);
    result
}

/// Create a family's worker venv, surviving a broken ensurepip.
/// Plain `python -m venv` fails on some Pythons at ensurepip — Colab,
/// 2026-11-06. On a venv failure this removes the half-built dir,
/// retries with --without-pip, and bootstraps pip via get-pip.py.
/// If any step fails the half-built dir is removed again and a
/// WorkerEnvError names venv creation as the problem. Returns the
/// venv python.
fn create_venv(family: &str, dir: &Path, runner: &dyn Runner) -> Result<PathBuf, WorkerEnvError> {
    let root = venv_path_in(dir, family);
    let sys_python = system_python().map_err(|e| {
        WorkerEnvError(format!("could not create worker env for {family}: {}", e.0))
    })?;
    let sys_py = sys_python.to_string_lossy().into_owned();
    let root_s = root.to_string_lossy().into_owned();
    let proc = runner
        .run(&[sys_py.clone(), "-m".into(), "venv".into(), root_s.clone()])
        .map_err(|exc| {
            WorkerEnvError(format!("could not create worker env for {family}: {exc}"))
        })?;
    if proc.code == 0 {
        return Ok(venv_python_in(dir, family));
    }
    // ensurepip (or the venv module) failed — retry without pip, then
    // bootstrap it. Remove the half-built env first: a venv dir with a
    // python but no packages is what turns this into a misleading
    // "missing packages" refusal on the next run.
    remove_dir_all(&root);
    let proc = runner
        .run(&[
            sys_py.clone(),
            "-m".into(),
            "venv".into(),
            "--without-pip".into(),
            root_s.clone(),
        ])
        .map_err(|exc| {
            WorkerEnvError(format!("could not create worker env for {family}: {exc}"))
        })?;
    if proc.code != 0 {
        let detail = tail(&proc);
        let detail = if detail.is_empty() {
            "no output".to_string()
        } else {
            detail
        };
        remove_dir_all(&root);
        return Err(WorkerEnvError(format!(
            "could not create worker env for {family}: this Python cannot \
             build venvs ({detail}). Workaround: use this Python itself \
             as the worker — symlink {} at it after \
             installing transformers, accelerate, bitsandbytes and \
             pyyaml (the Colab workaround, docs/FINDINGS.md §14).",
            venv_python_in(dir, family).display()
        )));
    }
    let python = venv_python_in(dir, family);
    if let Err(e) = bootstrap_pip(&python, &sys_python, runner) {
        remove_dir_all(&root);
        return Err(e);
    }
    Ok(python)
}

// -- public operations -------------------------------------------------------

/// One row of `list_envs`: path, built?, installed transformers version.
#[derive(Debug, Clone)]
pub struct EnvRow {
    pub family: String,
    pub path: PathBuf,
    pub built: bool,
    pub transformers: Option<String>,
}

/// One row per family: path, built?, installed transformers version.
/// Mirrors `ollada.workers.list_envs`.
pub fn list_envs(recipes: &[Recipe], dir: &Path, runner: &dyn Runner) -> Vec<EnvRow> {
    let mut families: Vec<String> = recipes.iter().map(|r| r.family.clone()).collect();
    families.sort();
    families.dedup();
    families
        .iter()
        .map(|family| {
            let python = venv_python_in(dir, family);
            let built = python.exists();
            EnvRow {
                family: family.clone(),
                path: venv_path_in(dir, family),
                built,
                transformers: if built {
                    installed_transformers(&python, runner)
                } else {
                    None
                },
            }
        })
        .collect()
}

/// Return the venv python for a family's worker env, building if needed.
/// Mirrors `ollada.workers.ensure_env`. `hardware`, when given, is used
/// instead of a live detection (the tests inject it; the CLI passes None).
/// Prints `building worker env for <family> (one-time download) ...` to
/// stdout on a fresh build, exactly like Python.
pub fn ensure_env(
    recipe: &Recipe,
    catalog: &[Recipe],
    dir: &Path,
    runner: &dyn Runner,
    hardware: Option<Hardware>,
) -> Result<PathBuf, WorkerEnvError> {
    let family = recipe.family.clone();
    let python = venv_python_in(dir, &family);
    let spec = env_transformers(recipe);
    if python.exists() {
        if !spec.is_empty() {
            if let Some(installed) = installed_transformers(&python, runner) {
                if !spec_satisfied(&spec, &installed) {
                    return Err(WorkerEnvError(format!(
                        "worker env for {family} has transformers {installed}, \
                         but {} pins transformers{spec}. Rebuild it: \
                         ollada workers rebuild {}",
                        recipe.name, recipe.name
                    )));
                }
            }
        }
        // An env built before the union rule (or by hand) can pass the
        // pin check while missing a package another family model needs
        // — the Dell sdar env had transformers but no einops. Refuse
        // with the same rebuild pointer instead of a bare ImportError.
        if let Some(dists) = installed_distributions(&python, runner) {
            let needed: HashSet<String> = pip_packages(recipe, catalog)
                .iter()
                .map(|p| norm_dist_name(&pkg_key(p)))
                .collect();
            let mut missing: Vec<String> = needed
                .iter()
                .filter(|n| !dists.contains(*n))
                .cloned()
                .collect();
            missing.sort();
            if !missing.is_empty() {
                return Err(WorkerEnvError(format!(
                    "worker env for {family} is missing {}. Rebuild it: \
                     ollada workers rebuild {}",
                    missing.join(", "),
                    recipe.name
                )));
            }
        }
        return Ok(python);
    }
    // Fresh build.
    println!("building worker env for {family} (one-time download) ...");
    let python = create_venv(&family, dir, runner)?;
    let pkgs = pip_packages(recipe, catalog);
    // Fresh detection at build time: never inherit a stale cached
    // result from an earlier call in this process.
    let hardware = hardware.unwrap_or_else(detect_hardware);
    let torch_pkgs: Vec<String> = pkgs
        .iter()
        .filter(|p| p.as_str() == "torch")
        .cloned()
        .collect();
    let rest: Vec<String> = pkgs
        .iter()
        .filter(|p| p.as_str() != "torch")
        .cloned()
        .collect();
    if hardware.gpu_name.is_none() {
        // No NVIDIA GPU: torch from the CPU index, everything else from
        // PyPI (a single --index-url would move all packages, not just torch).
        let mut argv = vec![
            python.to_string_lossy().into_owned(),
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--index-url".into(),
            CPU_TORCH_INDEX.into(),
        ];
        argv.extend(torch_pkgs);
        run_checked(
            runner,
            &argv,
            &format!("install torch (CPU index) for {family}"),
        )?;
        if !rest.is_empty() {
            let mut argv = vec![
                python.to_string_lossy().into_owned(),
                "-m".into(),
                "pip".into(),
                "install".into(),
            ];
            argv.extend(rest);
            run_checked(
                runner,
                &argv,
                &format!("install worker packages for {family}"),
            )?;
        }
    } else {
        // GPU machine: torch from PyTorch's CUDA index, everything
        // else from PyPI — plain PyPI torch is CPU-only.
        let mut argv = vec![
            python.to_string_lossy().into_owned(),
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--index-url".into(),
            cuda_torch_index(&hardware),
        ];
        argv.extend(torch_pkgs);
        run_checked(
            runner,
            &argv,
            &format!("install torch (CUDA index) for {family}"),
        )?;
        if !rest.is_empty() {
            let mut argv = vec![
                python.to_string_lossy().into_owned(),
                "-m".into(),
                "pip".into(),
                "install".into(),
            ];
            argv.extend(rest);
            run_checked(
                runner,
                &argv,
                &format!("install worker packages for {family}"),
            )?;
        }
    }
    // Verify the installed torch matches the detected hardware. A
    // GPU machine whose torch reports no CUDA is a broken env, not a
    // slow one — fail loudly here instead of surfacing minutes later
    // as CPU generation (the desktop dream env did exactly that).
    if hardware.gpu_name.is_some() && !worker_torch_has_cuda(&python, runner) {
        return Err(WorkerEnvError(format!(
            "worker env for {family} was built on a GPU machine \
             ({}), but its torch reports no CUDA. \
             The torch install did not pick up CUDA — rebuild it: \
             ollada workers rebuild {}",
            hardware.gpu_name.clone().unwrap_or_default(),
            recipe.name
        )));
    }
    // Guard against incompatible family pins: pip resolved the requested
    // model's pin, but a sibling model may need a disjoint range. Fail
    // loudly at build time rather than thrashing between rebuilds.
    if let Some(installed) = installed_transformers(&python, runner) {
        for sibling in family_recipes(&family, catalog) {
            let sibling_spec = env_transformers(sibling);
            if !sibling_spec.is_empty() && !spec_satisfied(&sibling_spec, &installed) {
                return Err(WorkerEnvError(format!(
                    "worker env for {family} resolved transformers {installed}, \
                     which does not satisfy {}'s pin \
                     transformers{sibling_spec} — this family needs \
                     per-model envs, which are not built yet",
                    sibling.name
                )));
            }
        }
    }
    Ok(python)
}

/// Delete a family's worker env and rebuild it from the recipe.
/// Mirrors `ollada.workers.rebuild_env`.
pub fn rebuild_env(
    recipe: &Recipe,
    catalog: &[Recipe],
    dir: &Path,
    runner: &dyn Runner,
    hardware: Option<Hardware>,
) -> Result<PathBuf, WorkerEnvError> {
    rebuild_env_with_remover(recipe, catalog, dir, runner, hardware, &|p| {
        remove_dir_all(p);
    })
}

/// `rebuild_env` with the deletion step injectable, so tests can
/// simulate the Windows file-locking case where the dir survives
/// (mirrors Python's test monkeypatching `shutil.rmtree` to a no-op).
pub fn rebuild_env_with_remover(
    recipe: &Recipe,
    catalog: &[Recipe],
    dir: &Path,
    runner: &dyn Runner,
    hardware: Option<Hardware>,
    remover: &dyn Fn(&Path),
) -> Result<PathBuf, WorkerEnvError> {
    remover(&venv_path_in(dir, &recipe.family));
    if venv_python_in(dir, &recipe.family).exists() {
        // Windows file locking (or a still-running worker) can leave
        // the env behind; proceeding would make ensure_env return the
        // old env untouched while this command reports success.
        return Err(WorkerEnvError(format!(
            "worker env for {} could not be deleted — \
             it is probably still in use. Stop any running ollada \
             process and rebuild again: \
             ollada workers rebuild {}",
            recipe.family, recipe.name
        )));
    }
    ensure_env(recipe, catalog, dir, runner, hardware)
}

#[cfg(test)]
type FakeBehavior = Box<dyn Fn(&[String]) -> Result<RunOutput, String> + Send + Sync>;

#[cfg(test)]
pub(crate) struct FakeRunner {
    calls: std::sync::Mutex<Vec<Vec<String>>>,
    behavior: FakeBehavior,
}

#[cfg(test)]
impl FakeRunner {
    pub(crate) fn new(
        behavior: impl Fn(&[String]) -> Result<RunOutput, String> + Send + Sync + 'static,
    ) -> Self {
        FakeRunner {
            calls: std::sync::Mutex::new(Vec::new()),
            behavior: Box::new(behavior),
        }
    }

    pub(crate) fn ok() -> Self {
        FakeRunner::new(|_| Ok(RunOutput::new(0, "", "")))
    }

    pub(crate) fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().unwrap().clone()
    }
}

#[cfg(test)]
impl Runner for FakeRunner {
    fn run(&self, argv: &[String]) -> Result<RunOutput, String> {
        self.calls.lock().unwrap().push(argv.to_vec());
        (self.behavior)(argv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::{load_catalog, validate_recipe};
    use serde_yaml::Value as YamlValue;
    use std::collections::HashMap;

    fn catalog() -> Vec<Recipe> {
        load_catalog(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("ollada")
                .join("catalog"),
        )
        .unwrap()
    }

    fn recipe(name: &str, catalog: &[Recipe]) -> Recipe {
        catalog.iter().find(|r| r.name == name).unwrap().clone()
    }

    fn make_recipe(name: &str, family: &str, spec: &str, extras: &[&str]) -> Recipe {
        let extras_yaml = extras
            .iter()
            .map(|e| format!("'{e}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let yaml = format!(
            "name: '{name}'\nfamily: {family}\nsource: org/x\nlicense: MIT\ntask: chat\n\
             description: ''\nsize_gb_bf16: 1.0\nquant_default: bf16\ndtype: float16\n\
             generate: {{steps: 1}}\nenv: {{transformers: '{spec}', extras: [{extras_yaml}]}}\nverified: {{}}\n"
        );
        let data: YamlValue = serde_yaml::from_str(&yaml).unwrap();
        validate_recipe(&data, None).unwrap()
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ollada-rs-workers-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn gpu_hw() -> Hardware {
        Hardware {
            gpu_name: Some("Test GPU".to_string()),
            vram_total_gb: Some(16.0),
            vram_free_gb: Some(15.0),
            ram_total_gb: Some(32.0),
        }
    }

    fn cpu_hw() -> Hardware {
        Hardware {
            gpu_name: None,
            ram_total_gb: Some(32.0),
            ..Default::default()
        }
    }

    /// Fake runner answering the metadata probes like a healthy env
    /// with the given transformers version, and CUDA as given.
    /// A venv call also materialises the venv python file, so the
    /// post-build `installed_transformers` probe sees a real path
    /// (Python's probe returns None for a missing python too).
    fn ok_runner(transformers_version: &'static str, cuda: &'static str) -> FakeRunner {
        FakeRunner::new(move |argv| {
            if argv.len() > 2 && argv[1] == "-m" && argv[2] == "venv" {
                if let Some(root) = argv.last() {
                    let root = PathBuf::from(root);
                    let python = if cfg!(windows) {
                        root.join("Scripts").join("python.exe")
                    } else {
                        root.join("bin").join("python")
                    };
                    if let Some(parent) = python.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&python, "");
                }
            }
            if argv.iter().any(|a| a.contains("torch.cuda.is_available()")) {
                return Ok(RunOutput::new(0, cuda, ""));
            }
            if argv.iter().any(|a| a.contains("m.version('transformers')")) {
                return Ok(RunOutput::new(0, transformers_version, ""));
            }
            if argv.iter().any(|a| a.contains("m.distributions()")) {
                return Ok(RunOutput::new(
                    0,
                    "pyyaml\ntorch\ntransformers\naccelerate\nbitsandbytes\neinops\n",
                    "",
                ));
            }
            Ok(RunOutput::new(0, "", ""))
        })
    }

    fn pip_calls(calls: &[Vec<String>]) -> Vec<&Vec<String>> {
        calls
            .iter()
            .filter(|c| c.iter().any(|a| a == "pip"))
            .collect()
    }

    fn venv_calls(calls: &[Vec<String>]) -> Vec<&Vec<String>> {
        calls
            .iter()
            .filter(|c| c.len() > 2 && c[1] == "-m" && c[2] == "venv")
            .collect()
    }

    // -- install set / torch index --------------------------------------------

    #[test]
    fn pip_packages_dedupe_and_pin() {
        let cat = catalog();
        assert_eq!(
            pip_packages(&recipe("dream:7b", &cat), &cat),
            vec![
                "pyyaml",
                "torch",
                "transformers>=4.53.3,<5",
                "accelerate",
                "bitsandbytes"
            ]
        );
        // sdar unions the family's extras: 1.7b alone would omit einops,
        // which 4b needs — the Dell failure. Both models get the same set.
        let sdar_17 = pip_packages(&recipe("sdar:1.7b", &cat), &cat);
        assert_eq!(
            sdar_17,
            vec![
                "pyyaml",
                "torch",
                "transformers>=4.57,<5",
                "accelerate",
                "einops"
            ]
        );
        assert_eq!(sdar_17, pip_packages(&recipe("sdar:4b", &cat), &cat));
        assert!(pip_packages(&recipe("diffcoder:8b", &cat), &cat)
            .contains(&"transformers==5.3.0".to_string()));
    }

    #[test]
    fn gb10_uses_cuda_130_index_and_other_gpus_keep_default() {
        assert_eq!(
            cuda_torch_index_with_override(
                &Hardware {
                    gpu_name: Some("NVIDIA GB10".to_string()),
                    ..Default::default()
                },
                None
            ),
            "https://download.pytorch.org/whl/cu130"
        );
        assert_eq!(
            cuda_torch_index_with_override(&gpu_hw(), None),
            "https://download.pytorch.org/whl/cu128"
        );
        assert_eq!(
            cuda_torch_index_with_override(
                &Hardware {
                    gpu_name: Some("NVIDIA GB10".to_string()),
                    ..Default::default()
                },
                Some("https://example.test/torch")
            ),
            "https://example.test/torch"
        );
        // An empty override is no override (Python: `if override:`).
        assert_eq!(
            cuda_torch_index_with_override(&gpu_hw(), Some("")),
            "https://download.pytorch.org/whl/cu128"
        );
    }

    #[test]
    fn venv_python_layout_is_platform_specific() {
        let dir = PathBuf::from("/tmp/w");
        if cfg!(windows) {
            assert_eq!(
                venv_python_in(&dir, "dream"),
                PathBuf::from("/tmp/w/dream/Scripts/python.exe")
            );
        } else {
            assert_eq!(
                venv_python_in(&dir, "dream"),
                PathBuf::from("/tmp/w/dream/bin/python")
            );
        }
    }

    #[test]
    fn system_python_is_found_on_path() {
        let py = system_python().unwrap();
        assert!(py.is_file(), "system python not found: {}", py.display());
    }

    // -- ensure_env: fresh builds ----------------------------------------------

    #[test]
    fn ensure_env_builds_with_gpu_index_on_gpu_machine() {
        let cat = catalog();
        let dir = tmp_dir("gpu-build");
        let runner = ok_runner("4.57.6", "True");
        let python = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap();
        assert_eq!(python, venv_python_in(&dir, "dream"));
        let calls = runner.calls();
        assert_eq!(venv_calls(&calls)[0][1], "-m");
        assert_eq!(venv_calls(&calls)[0][2], "venv");
        let pip = pip_calls(&calls);
        assert_eq!(pip.len(), 2);
        let idx = pip[0].iter().position(|a| a == "--index-url").unwrap();
        assert_eq!(pip[0][idx + 1], "https://download.pytorch.org/whl/cu128");
        assert_eq!(pip[0].last().unwrap(), "torch");
        assert!(!pip[1].contains(&"torch".to_string()));
        assert!(pip[1].contains(&"transformers>=4.53.3,<5".to_string()));
    }

    #[test]
    fn ensure_env_splits_torch_to_cpu_index_without_gpu() {
        let cat = catalog();
        let dir = tmp_dir("cpu-build");
        let runner = ok_runner("4.57.6", "True");
        ensure_env(
            &recipe("sdar:4b", &cat),
            &cat,
            &dir,
            &runner,
            Some(cpu_hw()),
        )
        .unwrap();
        let calls = runner.calls();
        let pip = pip_calls(&calls);
        assert_eq!(pip.len(), 2);
        let idx = pip[0].iter().position(|a| a == "--index-url").unwrap();
        assert_eq!(pip[0][idx + 1], "https://download.pytorch.org/whl/cpu");
        assert_eq!(pip[0].last().unwrap(), "torch");
        assert!(!pip[1].contains(&"torch".to_string()));
    }

    #[test]
    fn ensure_env_gpu_build_refuses_when_installed_torch_has_no_cuda() {
        let cat = catalog();
        let dir = tmp_dir("no-cuda");
        let runner = ok_runner("4.57.6", "False");
        let err = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap_err();
        assert_eq!(
            err.0,
            concat!(
                "worker env for dream was built on a GPU machine (Test GPU), but its torch reports no CUDA. ",
                "The torch install did not pick up CUDA — rebuild it: ollada workers rebuild dream:7b"
            )
        );
    }

    #[test]
    fn ensure_env_sibling_pin_guard_fails_loudly() {
        // Synthetic family whose two recipes pin disjoint transformers
        // ranges; pip resolves the requested one, the sibling then fails.
        let cat = vec![
            make_recipe("a:1b", "fam", "==5.3.0", &[]),
            make_recipe("b:1b", "fam", "==4.57.6", &[]),
        ];
        let dir = tmp_dir("sibling-guard");
        let runner = ok_runner("5.3.0", "True");
        let err = ensure_env(&cat[0], &cat, &dir, &runner, Some(cpu_hw())).unwrap_err();
        assert_eq!(
            err.0,
            concat!(
                "worker env for fam resolved transformers 5.3.0, which does not satisfy b:1b's pin ",
                "transformers==4.57.6 — this family needs per-model envs, which are not built yet"
            )
        );
    }

    // -- ensure_env: existing envs ----------------------------------------------

    fn touch_venv_python(dir: &Path, family: &str) -> PathBuf {
        let python = venv_python_in(dir, family);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, "").unwrap();
        python
    }

    #[test]
    fn ensure_env_refuses_pin_mismatch() {
        let cat = catalog();
        let dir = tmp_dir("pin-mismatch");
        touch_venv_python(&dir, "dream");
        let runner = ok_runner("5.3.0", "True");
        let err = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap_err();
        // Full string, byte-for-byte against the Python message
        // captured live (fake venv python reporting 5.3.0).
        assert_eq!(
            err.0,
            concat!(
                "worker env for dream has transformers 5.3.0, but dream:7b pins transformers>=4.53.3,<5. ",
                "Rebuild it: ollada workers rebuild dream:7b"
            )
        );
    }

    #[test]
    fn ensure_env_missing_package_refuses_with_rebuild_pointer() {
        let cat = catalog();
        let dir = tmp_dir("missing-pkg");
        touch_venv_python(&dir, "sdar");
        // The Dell sdar env: right pin, but built from the 1.7b recipe,
        // so einops was missing when 4b ran.
        let runner = FakeRunner::new(|argv| {
            if argv.iter().any(|a| a.contains("m.version('transformers')")) {
                return Ok(RunOutput::new(0, "4.57.6", ""));
            }
            if argv.iter().any(|a| a.contains("m.distributions()")) {
                return Ok(RunOutput::new(
                    0,
                    "pyyaml\ntorch\ntransformers\naccelerate\n",
                    "",
                ));
            }
            Ok(RunOutput::new(0, "", ""))
        });
        let err = ensure_env(
            &recipe("sdar:4b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap_err();
        // Full string, byte-for-byte against the Python message
        // captured live (fake venv python, distributions without einops).
        assert_eq!(
            err.0,
            "worker env for sdar is missing einops. Rebuild it: ollada workers rebuild sdar:4b"
        );
    }

    #[test]
    fn ensure_env_matching_pin_returns_existing_python() {
        let cat = catalog();
        let dir = tmp_dir("existing-ok");
        let python = touch_venv_python(&dir, "dream");
        let runner = ok_runner("4.57.6", "True");
        assert_eq!(
            ensure_env(
                &recipe("dream:7b", &cat),
                &cat,
                &dir,
                &runner,
                Some(gpu_hw())
            )
            .unwrap(),
            python
        );
        assert!(venv_calls(&runner.calls()).is_empty());
    }

    // -- venv creation failure paths --------------------------------------------

    #[test]
    fn ensure_env_recovers_when_venv_ensurepip_fails() {
        // Colab 2026-11-06: `python -m venv` dies at ensurepip, leaving
        // a half-built env; the build must retry with --without-pip and
        // bootstrap pip, not die the same way.
        let cat = catalog();
        let dir = tmp_dir("ensurepip-recovery");
        let runner = FakeRunner::new(|argv| {
            if argv.len() > 2
                && argv[1] == "-m"
                && argv[2] == "venv"
                && !argv.contains(&"--without-pip".to_string())
            {
                return Ok(RunOutput::new(1, "", "ensurepip failed"));
            }
            if argv.iter().any(|a| a.contains("torch.cuda.is_available()")) {
                return Ok(RunOutput::new(0, "True", ""));
            }
            if argv.iter().any(|a| a.contains("m.version('transformers')")) {
                return Ok(RunOutput::new(0, "4.57.6", ""));
            }
            Ok(RunOutput::new(0, "", ""))
        });
        let python = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap();
        assert_eq!(python, venv_python_in(&dir, "dream"));
        let calls = runner.calls();
        let venv = venv_calls(&calls);
        assert_eq!(venv.len(), 2);
        assert_eq!(venv[0][1], "-m");
        assert_eq!(venv[0][2], "venv");
        assert!(venv[1].contains(&"--without-pip".to_string()));
        // get-pip.py ran with the venv python before any pip install.
        let getpip: Vec<&Vec<String>> = calls
            .iter()
            .filter(|c| c.last().map(|a| a.ends_with("get-pip.py")).unwrap_or(false))
            .collect();
        assert_eq!(getpip.len(), 1);
        assert_eq!(getpip[0][0], python.to_string_lossy().into_owned());
        let getpip_pos = calls.iter().position(|c| c == getpip[0]).unwrap();
        assert!(calls[getpip_pos + 1..]
            .iter()
            .any(|c| c.contains(&"pip".to_string())));
    }

    #[test]
    fn ensure_env_venv_failure_twice_raises_and_cleans_up() {
        let cat = catalog();
        let dir = tmp_dir("venv-fails-twice");
        let dir2 = dir.clone();
        let runner = FakeRunner::new(move |argv| {
            if argv.len() > 2 && argv[1] == "-m" && argv[2] == "venv" {
                // Leave a half-built dir behind, like the real failure.
                let root = venv_path_in(&dir2, "dream");
                let _ = std::fs::create_dir_all(root.join("bin"));
                let _ = std::fs::write(root.join("bin").join("python"), "");
            }
            Ok(RunOutput::new(1, "", "ensurepip failed"))
        });
        let err = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap_err();
        assert!(
            err.0.contains("cannot build venvs"),
            "unexpected: {}",
            err.0
        );
        assert!(err.0.contains("Colab workaround"));
        assert!(!venv_path_in(&dir, "dream").exists());
    }

    #[test]
    fn ensure_env_pip_bootstrap_failure_cleans_up() {
        let cat = catalog();
        let dir = tmp_dir("bootstrap-fails");
        let dir2 = dir.clone();
        let runner = FakeRunner::new(move |argv| {
            if argv.len() > 2 && argv[1] == "-m" && argv[2] == "venv" {
                if !argv.contains(&"--without-pip".to_string()) {
                    return Ok(RunOutput::new(1, "", "ensurepip failed"));
                }
                let root = venv_path_in(&dir2, "dream");
                let _ = std::fs::create_dir_all(root.join("bin"));
                let _ = std::fs::write(root.join("bin").join("python"), "");
                return Ok(RunOutput::new(0, "", ""));
            }
            if argv.iter().any(|a| a.contains("urlretrieve")) {
                return Ok(RunOutput::new(1, "", "no network"));
            }
            Ok(RunOutput::new(0, "", ""))
        });
        let err = ensure_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap_err();
        assert!(err.0.contains("bootstrap pip"), "unexpected: {}", err.0);
        assert!(!venv_path_in(&dir, "dream").exists());
    }

    #[test]
    fn create_venv_with_real_python() {
        // The one live subprocess path: a real `python -m venv` with
        // the real runner. No pip installs, no network.
        let dir = tmp_dir("real-venv");
        let runner = RealRunner;
        let python = create_venv("dream", &dir, &runner).unwrap();
        assert!(python.is_file());
        let out = runner
            .run(&[
                python.to_string_lossy().into_owned(),
                "--version".to_string(),
            ])
            .unwrap();
        assert_eq!(out.code, 0);
        assert!(out.stdout.contains("Python 3"));
    }

    // -- rebuild / list ----------------------------------------------------------

    #[test]
    fn rebuild_deletes_then_builds() {
        let cat = catalog();
        let dir = tmp_dir("rebuild-ok");
        let marker = venv_path_in(&dir, "dream").join("marker.txt");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, "old").unwrap();
        let runner = ok_runner("4.57.6", "True");
        rebuild_env(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
        )
        .unwrap();
        assert!(!marker.exists());
        assert!(!venv_calls(&runner.calls()).is_empty());
    }

    #[test]
    fn rebuild_refuses_when_env_survives_deletion() {
        // Windows file locking can leave the env behind while rmtree
        // hides it; proceeding would return the old env untouched
        // while reporting success.
        let cat = catalog();
        let dir = tmp_dir("rebuild-survives");
        touch_venv_python(&dir, "dream");
        let runner = ok_runner("4.57.6", "True");
        let err = rebuild_env_with_remover(
            &recipe("dream:7b", &cat),
            &cat,
            &dir,
            &runner,
            Some(gpu_hw()),
            &|_| {}, // no-op remover: the env survives
        )
        .unwrap_err();
        assert_eq!(
            err.0,
            concat!(
                "worker env for dream could not be deleted — it is probably still in use. Stop any running ollada ",
                "process and rebuild again: ollada workers rebuild dream:7b"
            )
        );
    }

    #[test]
    fn list_envs_reports_built_and_version() {
        let cat = catalog();
        let dir = tmp_dir("list-envs");
        let runner = ok_runner("4.57.6", "True");
        let rows: HashMap<String, EnvRow> = list_envs(&cat, &dir, &runner)
            .into_iter()
            .map(|r| (r.family.clone(), r))
            .collect();
        assert_eq!(rows.len(), 4);
        assert!(!rows["dream"].built);
        assert_eq!(rows["dream"].transformers, None);
        touch_venv_python(&dir, "dream");
        let rows: HashMap<String, EnvRow> = list_envs(&cat, &dir, &runner)
            .into_iter()
            .map(|r| (r.family.clone(), r))
            .collect();
        assert!(rows["dream"].built);
        assert_eq!(rows["dream"].transformers.as_deref(), Some("4.57.6"));
    }

    #[test]
    fn installed_transformers_returns_none_without_python() {
        let runner = FakeRunner::ok();
        assert_eq!(
            installed_transformers(&PathBuf::from("/nonexistent/python"), &runner),
            None
        );
        assert_eq!(
            installed_distributions(&PathBuf::from("/nonexistent/python"), &runner),
            None
        );
    }
}
