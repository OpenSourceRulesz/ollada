//! Model manager — one model loaded at a time, serial generation.
//! A port of `ollada/runtime/manager.py` (the worker half only).
//!
//! The manager spawns one worker subprocess per model
//! (`<venv python> -m ollada.worker`, see `ollada/worker.py`) and
//! talks to it over stdin/stdout with one JSON object per line.
//! Stdout is protocol-only; the worker's stderr is inherited, so its
//! logs go to the manager's stderr, never the protocol. One model is
//! loaded at a time; generation is serialised by the manager lock.
//! Killing the worker is the clean-VRAM-unload half of the design.
//!
//! Deliberate differences from Python, all documented:
//! - **No in-process path.** Python's `OLLADA_INPROCESS=1` escape
//!   hatch restores in-process adapter loading. The Rust manager
//!   cannot do that (adapters are Python), so a set
//!   `OLLADA_INPROCESS` is a clear refusal, never a silent fallback.
//! - **Startup failure kills the worker.** Python's `_start` raises
//!   on a bad ready line and leaves the spawned process in `_LIVE`
//!   until process exit. Rust kills it immediately, so a broken
//!   spawn never lingers.
//! - **A non-object first line is a clean error.** Python calls
//!   `.get("event")` on the parsed first line and crashes with
//!   `AttributeError` if it is a JSON array or scalar. Rust returns
//!   a `WorkerError` instead.
//! - **Terminate on Unix, kill on Windows.** Python's
//!   `proc.terminate()` is SIGTERM on Unix and TerminateProcess on
//!   Windows. Rust has no portable terminate, so Unix sends SIGTERM
//!   (via libc) and waits up to 10 s before SIGKILL; Windows calls
//!   `Child::kill()` (TerminateProcess) and waits. Same observable
//!   outcome: the worker is gone after unload.
//! - **No `capture_and_record`.** Python records a CUDA peak from
//!   its own in-process torch when a chat result carries no peak.
//!   The Rust manager has no torch; a worker-reported peak of
//!   `None` (CPU, no CUDA) simply records nothing — which is what
//!   Python's capture returns on a CPU box too.
//! - **The fit check's RAM floor applies.** Python's
//!   `fit.assess(...).blocking_reason()` blocks on a host-RAM floor
//!   as well as on VRAM; the Rust manager calls the same check, so
//!   a refused load leaves state exactly as it was, before any env
//!   is built or any model evicted.

use crate::fit::{assess, detect_hardware, Hardware};
use crate::measurements::{load_record, record_peak};
use crate::pyrepr::{py_json, py_str};
use crate::recipes::{get_recipe, load_catalog, Recipe, RecipeError};
use crate::workers::{ensure_env, RealRunner, WorkerEnvError};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, Once};
use std::time::{Duration, Instant};

/// Worker startup is imports only; a load takes minutes, startup must not.
pub const READY_TIMEOUT: Duration = Duration::from_secs(120);
/// Terminate, wait this long, then kill.
pub const KILL_TIMEOUT: Duration = Duration::from_secs(10);

/// Mirrors `ollada.worker.WorkerError`: the protocol itself broke
/// (not a model failure).
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerError(pub String);

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for WorkerError {}

/// Mirrors `ollada.adapters.base.AdapterError`: the worker reported
/// a failure (`ok: false`); the message carries the worker's error,
/// plus its traceback when present.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterError(pub String);

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AdapterError {}

/// Mirrors `ollada.fit.FitError`: the fit check refused a load.
#[derive(Debug, Clone, PartialEq)]
pub struct FitError(pub String);

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FitError {}

/// Every way a manager operation can fail, keeping Python's error
/// classes distinct so callers (the CLI, the future server) can map
/// them the way Python's `main()` does: `FitError` prints
/// `fit check: ...` and exits 2; the rest print `error: ...`.
#[derive(Debug)]
pub enum ManagerError {
    Worker(WorkerError),
    Adapter(AdapterError),
    Fit(FitError),
    Recipe(RecipeError),
    Env(WorkerEnvError),
    /// `OLLADA_INPROCESS` is set: refused, never silently fallen back.
    InProcess(String),
}

impl std::fmt::Display for ManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManagerError::Worker(e) => write!(f, "{e}"),
            ManagerError::Adapter(e) => write!(f, "{e}"),
            ManagerError::Fit(e) => write!(f, "{e}"),
            ManagerError::Recipe(e) => write!(f, "{e}"),
            ManagerError::Env(e) => write!(f, "{e}"),
            ManagerError::InProcess(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ManagerError {}

impl From<WorkerError> for ManagerError {
    fn from(e: WorkerError) -> Self {
        ManagerError::Worker(e)
    }
}

impl From<AdapterError> for ManagerError {
    fn from(e: AdapterError) -> Self {
        ManagerError::Adapter(e)
    }
}

impl From<FitError> for ManagerError {
    fn from(e: FitError) -> Self {
        ManagerError::Fit(e)
    }
}

impl From<RecipeError> for ManagerError {
    fn from(e: RecipeError) -> Self {
        ManagerError::Recipe(e)
    }
}

impl From<WorkerEnvError> for ManagerError {
    fn from(e: WorkerEnvError) -> Self {
        ManagerError::Env(e)
    }
}

pub type ManagerResult<T> = Result<T, ManagerError>;

/// Mirrors `ollada.adapters.base.GenResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct GenResult {
    pub text: String,
    pub tokens: u64,
    pub seconds: f64,
    pub nfe: Option<u64>,
    pub peak_gb: Option<f64>,
    pub hardware: Option<String>,
}

/// What a worker's `load` reply carries.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadResult {
    pub loaded: Option<String>,
    pub family: Option<String>,
    pub supports_watch: bool,
    pub peak_gb: Option<f64>,
    pub hardware: Option<String>,
}

/// What `ensure_loaded` returns: the loaded model's identity plus
/// the worker's watch support, so callers need not reach into the adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadInfo {
    pub name: String,
    pub family: String,
    pub supports_watch: bool,
    pub peak_gb: Option<f64>,
    pub hardware: Option<String>,
}

/// What `status()` returns: `{"loaded": name|null, "family": ...}`.
#[allow(dead_code)] // ported API surface; the server phase calls this
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ManagerStatus {
    pub loaded: Option<String>,
    pub family: Option<String>,
}

/// Build (or reuse) a recipe's worker env. Called BEFORE evicting
/// the resident model, so a build failure never costs the loaded
/// model. The default is the real `workers::ensure_env`.
pub type EnvBuilder =
    Arc<dyn Fn(&Recipe, &[Recipe], &Path) -> Result<PathBuf, WorkerEnvError> + Send + Sync>;

/// Build a worker's argv from its venv python. The default is
/// `[python, -m, ollada.worker]`; tests inject a fake worker.
pub type ArgvBuilder = Arc<dyn Fn(&Path) -> Vec<String> + Send + Sync>;

/// One chat message. The worker protocol takes a list of
/// `{"role": ..., "content": ...}` dicts; the server phase will map
/// its JSON to these.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }
}

/// A step callback: `(pieces, block_i, num_blocks, step_i, steps)`,
/// each the worker's JSON value or `None` when the frame omits it —
/// mirrors Python's `msg.get(...)` forwarding.
pub type StepHandler = Box<
    dyn FnMut(
            Option<JsonValue>,
            Option<JsonValue>,
            Option<JsonValue>,
            Option<JsonValue>,
            Option<JsonValue>,
        ) + Send,
>;

// -- live-worker registry (atexit equivalent) --------------------------------

/// Pids of live workers. Python keeps the `Popen` objects in `_LIVE`
/// and terminates them from an `atexit` hook; Rust keeps pids and
/// kills by pid from a `libc::atexit` handler registered on first spawn.
static LIVE: std::sync::LazyLock<Mutex<HashSet<u32>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashSet::new()));
static REGISTER_ATEXIT: Once = Once::new();

fn register_live(pid: u32) {
    if let Ok(mut live) = LIVE.lock() {
        live.insert(pid);
    }
    REGISTER_ATEXIT.call_once(|| unsafe {
        libc::atexit(kill_all_live_atexit);
    });
}

fn deregister_live(pid: u32) {
    if let Ok(mut live) = LIVE.lock() {
        live.remove(&pid);
    }
}

extern "C" fn kill_all_live_atexit() {
    kill_all_live();
}

/// Terminate all live workers, wait up to 10 s, then kill stragglers.
/// Mirrors Python's `_kill_all_live`. Called at process exit via
/// `atexit`, and directly by tests.
pub fn kill_all_live() {
    let pids: Vec<u32> = match LIVE.lock() {
        Ok(mut live) => live.drain().collect(),
        Err(_) => return,
    };
    if pids.is_empty() {
        return;
    }
    #[cfg(unix)]
    {
        for pid in &pids {
            unsafe {
                libc::kill(*pid as i32, libc::SIGTERM);
            }
        }
        let deadline = Instant::now() + KILL_TIMEOUT;
        loop {
            if pids.iter().all(|pid| !pid_alive(*pid)) {
                return;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        for pid in &pids {
            if pid_alive(*pid) {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGKILL);
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        // No SIGTERM on Windows: sysinfo kills by pid
        // (TerminateProcess, like Child::kill).
        let mut sys = match sysinfo::System::new() {
            Ok(sys) => sys,
            Err(_) => return,
        };
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        for pid in pids {
            if let Some(process) = sys.process(sysinfo::Pid::from_u32(pid)) {
                let _ = process.kill();
            }
        }
    }
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        // A zombie is dead but unreaped (its parent still holds the
        // Child handle); kill(pid, 0) still reports it alive, so
        // read its state from /proc instead. The state letter
        // follows the last ") " (comm is parenthesised and may
        // contain spaces); Z/X mean gone.
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            if let Some(state) = stat.rsplit(") ").next().and_then(|s| s.chars().next()) {
                return state != 'Z' && state != 'X';
            }
        }
        // /proc entry gone (already reaped): not alive.
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        let rc = unsafe { libc::kill(pid as i32, 0) };
        if rc == 0 {
            return true;
        }
        // EPERM means the process exists but is not ours.
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

/// Format a process exit the way Python's `Popen.poll()` prints it:
/// the exit code, `-<signal>` on Unix, or `None` while still running.
fn rc_string(status: Option<std::process::ExitStatus>) -> String {
    match status {
        None => "None".to_string(),
        Some(s) => match s.code() {
            Some(code) => code.to_string(),
            #[cfg(unix)]
            None => match std::os::unix::process::ExitStatusExt::signal(&s) {
                Some(sig) => format!("-{sig}"),
                None => "None".to_string(),
            },
            #[cfg(not(unix))]
            None => "None".to_string(),
        },
    }
}

/// First 80 characters of a line, like Python's `line[:80]`.
fn first_80(line: &str) -> String {
    line.chars().take(80).collect()
}

// -- WorkerAdapter -------------------------------------------------------------

/// Manager-side handle for one worker subprocess. Mirrors Python's
/// `WorkerAdapter`: it speaks the JSON-lines protocol, so the manager
/// treats a worker like an adapter. The model itself lives in the
/// worker; `loaded` tracks the worker's state instead.
pub struct WorkerAdapter {
    recipe: Recipe,
    argv: Vec<String>,
    ollada_root: Option<PathBuf>,
    proc: Option<Child>,
    lines_rx: Option<mpsc::Receiver<Option<String>>>,
    loaded_flag: bool,
    next_id: u64,
    step_handler: Option<StepHandler>,
    supports_watch: bool,
    last_load: Option<LoadResult>,
    ready_timeout: Duration,
    kill_timeout: Duration,
}

impl WorkerAdapter {
    /// Production constructor: spawn `<worker_python> -m ollada.worker`
    /// with `PYTHONPATH` pointing at `ollada_root` (the manager's
    /// ollada location, so the worker runs the manager's exact code).
    pub fn new(recipe: Recipe, worker_python: &Path, ollada_root: &Path) -> Self {
        WorkerAdapter {
            recipe,
            argv: vec![
                worker_python.to_string_lossy().into_owned(),
                "-m".to_string(),
                "ollada.worker".to_string(),
            ],
            ollada_root: Some(ollada_root.to_path_buf()),
            proc: None,
            lines_rx: None,
            loaded_flag: false,
            next_id: 0,
            step_handler: None,
            supports_watch: false,
            last_load: None,
            ready_timeout: READY_TIMEOUT,
            kill_timeout: KILL_TIMEOUT,
        }
    }

    /// Test seam: spawn an arbitrary command (a fake worker script)
    /// instead of `-m ollada.worker`; no `PYTHONPATH` is set.
    #[cfg(test)]
    pub fn with_argv(recipe: Recipe, argv: Vec<String>) -> Self {
        WorkerAdapter {
            recipe,
            argv,
            ollada_root: None,
            proc: None,
            lines_rx: None,
            loaded_flag: false,
            next_id: 0,
            step_handler: None,
            supports_watch: false,
            last_load: None,
            ready_timeout: READY_TIMEOUT,
            kill_timeout: KILL_TIMEOUT,
        }
    }

    /// Test seam: shorten the ready/kill timeouts (production is 120 s / 10 s).
    #[cfg(test)]
    pub fn with_timeouts(mut self, ready: Duration, kill: Duration) -> Self {
        self.ready_timeout = ready;
        self.kill_timeout = kill;
        self
    }

    pub fn recipe_name(&self) -> &str {
        &self.recipe.name
    }

    pub fn recipe_family(&self) -> &str {
        &self.recipe.family
    }

    pub fn supports_watch(&self) -> bool {
        self.supports_watch
    }

    pub fn last_load(&self) -> Option<&LoadResult> {
        self.last_load.as_ref()
    }

    /// Mirrors Python's `loaded` property: the load succeeded and
    /// the worker process is still alive.
    pub fn loaded(&mut self) -> bool {
        if !self.loaded_flag {
            return false;
        }
        match self.proc.as_mut() {
            Some(proc) => matches!(proc.try_wait(), Ok(None)),
            None => false,
        }
    }

    // -- process plumbing -------------------------------------------------

    fn reader_loop(stdout: ChildStdout, tx: mpsc::Sender<Option<String>>) {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    if tx.send(Some(line)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(None); // EOF sentinel: the worker is gone
    }

    fn start(&mut self) -> ManagerResult<()> {
        let mut cmd = Command::new(&self.argv[0]);
        cmd.args(&self.argv[1..]);
        if let Some(root) = &self.ollada_root {
            // PYTHONPATH exists only so the worker finds the manager's
            // exact ollada code; the worker strips these entries from
            // sys.path once ollada is imported.
            let sep = if cfg!(windows) { ";" } else { ":" };
            let existing = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env(
                "PYTHONPATH",
                format!("{}{sep}{existing}", root.to_string_lossy()),
            );
        }
        let mut proc = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // worker logs go to the manager's stderr
            .spawn()
            .map_err(|e| WorkerError(format!("could not spawn worker ({}): {e}", self.argv[0])))?;
        let stdout = proc.stdout.take().expect("stdout was piped");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || Self::reader_loop(stdout, tx));
        self.lines_rx = Some(rx);
        self.proc = Some(proc);
        register_live(self.proc.as_ref().unwrap().id());
        let started = self.wait_ready();
        if let Err(e) = started {
            // Deliberate difference from Python (see module docs): a
            // broken spawn is killed here, not left in _LIVE.
            self.kill();
            return Err(e);
        }
        Ok(())
    }

    fn wait_ready(&mut self) -> ManagerResult<()> {
        let rx = self.lines_rx.as_ref().unwrap();
        let first = match rx.recv_timeout(self.ready_timeout) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(WorkerError(format!(
                    "worker did not print its ready line (is {} a working venv python?)",
                    self.argv[0]
                ))
                .into());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => None,
        };
        let Some(first) = first else {
            return Err(WorkerError(format!(
                "worker exited during startup (rc={})",
                self.poll_rc()
            ))
            .into());
        };
        let ready: JsonValue = match serde_json::from_str(&first) {
            Ok(v) => v,
            Err(_) => {
                return Err(WorkerError(format!(
                    "worker printed a non-JSON first line: {}",
                    py_str(&first_80(&first))
                ))
                .into());
            }
        };
        match ready.as_object() {
            Some(obj) if obj.get("event").and_then(|v| v.as_str()) == Some("ready") => Ok(()),
            _ => Err(WorkerError(format!(
                "worker first line was not a ready event: {}",
                py_json(&ready)
            ))
            .into()),
        }
    }

    fn poll_rc(&mut self) -> String {
        // EOF on the protocol pipe means the worker is dead or
        // dying; give the reaper a brief chance so the reported rc
        // is the real exit code, not a still-running "None".
        for _ in 0..10 {
            match self.proc.as_mut() {
                Some(proc) => match proc.try_wait() {
                    Ok(Some(status)) => return rc_string(Some(status)),
                    Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                    Err(_) => return "None".to_string(),
                },
                None => return "None".to_string(),
            }
        }
        match self.proc.as_mut() {
            Some(proc) => rc_string(proc.try_wait().ok().flatten()),
            None => "None".to_string(),
        }
    }

    fn send(&mut self, obj: &JsonValue) -> ManagerResult<()> {
        let line = serde_json::to_string(obj).expect("request is JSON-serialisable");
        let result = {
            let proc = self.proc.as_mut().expect("send with no process");
            let stdin = proc.stdin.as_mut().expect("stdin was piped");
            stdin
                .write_all(line.as_bytes())
                .and_then(|_| stdin.write_all(b"\n"))
                .and_then(|_| stdin.flush())
        };
        match result {
            Ok(()) => Ok(()),
            Err(_) => {
                Err(WorkerError(format!("worker stdin closed (rc={})", self.poll_rc())).into())
            }
        }
    }

    /// Send one request; return its result, forwarding step events.
    /// Mirrors Python's `_call`: only one request is outstanding at
    /// a time (the manager holds its lock), so no id correlation is
    /// needed beyond matching the response id.
    fn call(
        &mut self,
        op: &str,
        fields: serde_json::Map<String, JsonValue>,
    ) -> ManagerResult<JsonValue> {
        self.next_id += 1;
        let req_id = self.next_id;
        let mut obj = serde_json::Map::new();
        obj.insert("id".to_string(), JsonValue::from(req_id));
        obj.extend(fields);
        obj.insert("op".to_string(), JsonValue::String(op.to_string()));
        self.send(&JsonValue::Object(obj))?;
        loop {
            let line = {
                let rx = self.lines_rx.as_ref().expect("call with no reader");
                // Reader thread gone: the worker is gone.
                rx.recv().unwrap_or_default()
            };
            let Some(line) = line else {
                return Err(WorkerError(format!(
                    "worker exited mid-request (rc={})",
                    self.poll_rc()
                ))
                .into());
            };
            let msg: JsonValue = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => {
                    return Err(WorkerError(format!(
                        "worker printed a non-JSON line: {}",
                        py_str(&first_80(&line))
                    ))
                    .into());
                }
            };
            let msg = match msg.as_object() {
                Some(m) => m,
                None => {
                    return Err(WorkerError(format!(
                        "worker printed a non-object line: {}",
                        py_str(&first_80(&line))
                    ))
                    .into());
                }
            };
            if msg.get("event").and_then(|v| v.as_str()) == Some("step") {
                if let Some(handler) = self.step_handler.as_mut() {
                    handler(
                        msg.get("pieces").cloned(),
                        msg.get("block_i").cloned(),
                        msg.get("num_blocks").cloned(),
                        msg.get("step_i").cloned(),
                        msg.get("steps").cloned(),
                    );
                }
                continue;
            }
            if msg.get("id") != Some(&JsonValue::from(req_id)) {
                continue; // stale or unsolicited; the protocol has no other traffic
            }
            if !msg.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                let mut err = msg
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("worker error")
                    .to_string();
                if let Some(tb) = msg.get("traceback").and_then(|v| v.as_str()) {
                    err.push('\n');
                    err.push_str(tb);
                }
                return Err(AdapterError(err).into());
            }
            return Ok(msg.get("result").cloned().unwrap_or(JsonValue::Null));
        }
    }

    /// Terminate, wait up to the kill timeout, then kill; close
    /// stdin; remove from the live set. Mirrors Python's `_kill`.
    pub fn kill(&mut self) {
        let Some(mut proc) = self.proc.take() else {
            return;
        };
        let pid = proc.id();
        deregister_live(pid);
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
        #[cfg(not(unix))]
        let _ = proc.kill(); // Windows: no SIGTERM; TerminateProcess
        let deadline = Instant::now() + self.kill_timeout;
        loop {
            match proc.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = proc.kill();
                        let _ = proc.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }
        drop(proc.stdin.take()); // close stdin, after the wait, like Python
        drop(self.lines_rx.take());
    }

    // -- adapter interface -------------------------------------------------

    /// Spawn the worker and load the model in it. Returns the load result.
    pub fn load(&mut self) -> ManagerResult<LoadResult> {
        self.start()?;
        let mut fields = serde_json::Map::new();
        fields.insert(
            "model".to_string(),
            JsonValue::String(self.recipe.name.clone()),
        );
        let result = match self.call("load", fields) {
            Ok(r) => r,
            Err(e) => {
                self.kill();
                return Err(e);
            }
        };
        self.loaded_flag = true;
        let load = LoadResult {
            loaded: result
                .get("loaded")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            family: result
                .get("family")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            supports_watch: result
                .get("supports_watch")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            peak_gb: result.get("peak_gb").and_then(|v| v.as_f64()),
            hardware: result
                .get("hardware")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        };
        self.supports_watch = load.supports_watch;
        self.last_load = Some(load.clone());
        Ok(load)
    }

    /// Kill the worker (the clean-VRAM-unload half) and clear state.
    pub fn unload(&mut self) {
        self.kill();
        self.loaded_flag = false;
    }

    /// Generate one reply. `messages` go verbatim; `overrides` is
    /// sent as `null` when `None`, like Python.
    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        overrides: Option<JsonValue>,
        on_step: Option<StepHandler>,
    ) -> ManagerResult<GenResult> {
        if !self.loaded() {
            return Err(AdapterError("chat before load".to_string()).into());
        }
        let mut fields = serde_json::Map::new();
        fields.insert(
            "messages".to_string(),
            serde_json::to_value(messages).expect("messages serialise"),
        );
        fields.insert(
            "overrides".to_string(),
            overrides.unwrap_or(JsonValue::Null),
        );
        self.step_handler = on_step;
        let result = self.call("chat", fields);
        self.step_handler = None;
        let result = result?;
        let text = match result.get("text").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => {
                return Err(AdapterError("worker chat result has no 'text'".to_string()).into());
            }
        };
        Ok(GenResult {
            text,
            tokens: result.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            seconds: result
                .get("seconds")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            nfe: result.get("nfe").and_then(|v| v.as_u64()),
            peak_gb: result.get("peak_gb").and_then(|v| v.as_f64()),
            hardware: result
                .get("hardware")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }
}

impl Drop for WorkerAdapter {
    /// A dropped adapter must not leave a worker behind: killing
    /// here covers the load-failure path and any caller that forgets
    /// `unload()`. (The atexit registry is the backstop for a
    /// process that dies without dropping.)
    fn drop(&mut self) {
        self.kill();
    }
}

// -- ModelManager --------------------------------------------------------------

/// The ollada location for a catalog directory: the repo root, i.e.
/// the parent of the parent of `<root>/ollada/catalog`. Falls back
/// to walking up from the catalog looking for `ollada/worker.py`.
/// Mirrors Python's `Path(manager.py).parent.parent.parent`.
pub fn ollada_root_for_catalog(catalog_dir: &Path) -> PathBuf {
    if let Some(pkg) = catalog_dir.parent() {
        if let Some(root) = pkg.parent() {
            if root.join("ollada").join("worker.py").is_file() {
                return root.to_path_buf();
            }
        }
    }
    let mut dir = Some(catalog_dir.to_path_buf());
    while let Some(d) = dir {
        if d.join("ollada").join("worker.py").is_file() {
            return d;
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }
    catalog_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| catalog_dir.to_path_buf())
}

fn inprocess_from_env() -> bool {
    std::env::var("OLLADA_INPROCESS")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// One model loaded at a time, serial generation. Mirrors Python's
/// `ModelManager`; the state lives behind one mutex, which is also
/// the serial-generation lock (Python's `_lock` plus the adapter
/// slot, in one).
#[allow(dead_code)] // ported API surface; `status`/`unload`/
/// `ensure_loaded` are the server phase's entry points (the CLI
/// only uses `chat`)
pub struct ModelManager {
    state: Mutex<Option<WorkerAdapter>>,
    catalog_dir: PathBuf,
    workers_dir: PathBuf,
    meas_dir: PathBuf,
    /// Injected hardware for the fit check (`None` = live detect).
    hardware_override: Mutex<Option<Hardware>>,
    /// Injected in-process flag (`None` = read `OLLADA_INPROCESS`).
    /// A test seam: setting env vars from parallel tests is racy.
    inprocess_override: Mutex<Option<bool>>,
    /// Build (or reuse) the worker env for a recipe. The default is
    /// the real `workers::ensure_env` with the live runner; tests
    /// inject a fake. Called BEFORE evicting the resident model, so
    /// a build failure never costs the loaded model.
    env_builder: EnvBuilder,
    /// Build the worker's argv from its venv python. The default is
    /// `[python, -m, ollada.worker]`; tests inject a fake worker.
    argv_builder: ArgvBuilder,
    ready_timeout: Duration,
    kill_timeout: Duration,
}

// `status` / `unload` / `ensure_loaded` are the server phase's
// entry points (the CLI only uses `chat`); the test seams are used
// by the tests in this file.
#[allow(dead_code)]
impl ModelManager {
    pub fn new(catalog_dir: &Path, workers_dir: &Path, meas_dir: &Path) -> Self {
        ModelManager {
            state: Mutex::new(None),
            catalog_dir: catalog_dir.to_path_buf(),
            workers_dir: workers_dir.to_path_buf(),
            meas_dir: meas_dir.to_path_buf(),
            hardware_override: Mutex::new(None),
            inprocess_override: Mutex::new(None),
            env_builder: Arc::new(|recipe, catalog, dir| {
                ensure_env(recipe, catalog, dir, &RealRunner, None)
            }),
            argv_builder: Arc::new(|python| {
                vec![
                    python.to_string_lossy().into_owned(),
                    "-m".to_string(),
                    "ollada.worker".to_string(),
                ]
            }),
            ready_timeout: READY_TIMEOUT,
            kill_timeout: KILL_TIMEOUT,
        }
    }

    /// Test seam: inject hardware for the fit check.
    pub fn set_hardware_override(&self, hardware: Hardware) {
        if let Ok(mut slot) = self.hardware_override.lock() {
            *slot = Some(hardware);
        }
    }

    /// Test seam: inject the in-process flag instead of reading the env.
    pub fn set_inprocess_override(&self, value: bool) {
        if let Ok(mut slot) = self.inprocess_override.lock() {
            *slot = Some(value);
        }
    }

    /// Test seam: inject the env builder.
    pub fn set_env_builder(&mut self, builder: EnvBuilder) {
        self.env_builder = builder;
    }

    /// Test seam: inject the worker argv builder.
    pub fn set_argv_builder(&mut self, builder: ArgvBuilder) {
        self.argv_builder = builder;
    }

    /// Test seam: shorten the ready/kill timeouts.
    pub fn set_timeouts(&mut self, ready: Duration, kill: Duration) {
        self.ready_timeout = ready;
        self.kill_timeout = kill;
    }

    fn hardware(&self) -> Hardware {
        if let Ok(slot) = self.hardware_override.lock() {
            if let Some(hw) = slot.as_ref() {
                return hw.clone();
            }
        }
        detect_hardware()
    }

    fn inprocess(&self) -> bool {
        if let Ok(slot) = self.inprocess_override.lock() {
            if let Some(v) = *slot {
                return v;
            }
        }
        inprocess_from_env()
    }

    /// Mirrors Python's `status()`: `{"loaded": name|null,
    /// "family": family|null}`.
    pub fn status(&self) -> ManagerStatus {
        match self.state.lock() {
            Ok(state) => match state.as_ref() {
                Some(adapter) => ManagerStatus {
                    loaded: Some(adapter.recipe_name().to_string()),
                    family: Some(adapter.recipe_family().to_string()),
                },
                None => ManagerStatus {
                    loaded: None,
                    family: None,
                },
            },
            Err(_) => ManagerStatus {
                loaded: None,
                family: None,
            },
        }
    }

    pub fn current_name(&self) -> Option<String> {
        self.status().loaded
    }

    /// Load `name` if it is not already loaded, and return its load
    /// info. The order is Python's, and it matters: the fit refusal
    /// comes before any env build or eviction, and the env build
    /// comes before evicting the resident model.
    pub fn ensure_loaded(&self, name: &str, force: bool) -> ManagerResult<LoadInfo> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ManagerError::Worker(WorkerError("manager lock poisoned".to_string())))?;
        self.ensure_loaded_locked(&mut state, name, force)
    }

    fn ensure_loaded_locked(
        &self,
        state: &mut Option<WorkerAdapter>,
        name: &str,
        force: bool,
    ) -> ManagerResult<LoadInfo> {
        let recipe = get_recipe(name, &self.catalog_dir)?;
        if let Some(adapter) = state.as_mut() {
            if adapter.recipe_name() == recipe.name && adapter.loaded() {
                let last = adapter.last_load().cloned();
                let info = match last {
                    Some(load) => LoadInfo {
                        name: adapter.recipe_name().to_string(),
                        family: adapter.recipe_family().to_string(),
                        supports_watch: load.supports_watch,
                        peak_gb: load.peak_gb,
                        hardware: load.hardware,
                    },
                    None => LoadInfo {
                        name: adapter.recipe_name().to_string(),
                        family: adapter.recipe_family().to_string(),
                        supports_watch: adapter.supports_watch(),
                        peak_gb: None,
                        hardware: None,
                    },
                };
                return Ok(info);
            }
        }
        if !force {
            // Refuse BEFORE unloading the resident model or touching
            // the GPU: a refused load leaves the current state exactly
            // as it was. The fit check runs before any worker env is
            // built, so a refused model never pays the one-time env
            // download either.
            let hardware = self.hardware();
            let record = load_record(&recipe.name, &self.meas_dir);
            let reason = assess(&recipe, &hardware, record.as_ref()).blocking_reason();
            if let Some(reason) = reason {
                return Err(FitError(reason).into());
            }
        }
        if self.inprocess() {
            // Deliberate refusal (see module docs): the Rust manager
            // cannot load adapters in-process — adapters are Python.
            return Err(ManagerError::InProcess(
                "OLLADA_INPROCESS is set, but in-process loading is not supported \
                 by the Rust manager: adapters are Python, and the Rust manager can \
                 only talk to worker subprocesses. Unset OLLADA_INPROCESS to load \
                 through a worker instead."
                    .to_string(),
            ));
        }
        // Env build happens before evicting the resident model: a
        // build failure must not cost the currently loaded model.
        let catalog = load_catalog(&self.catalog_dir)?;
        let worker_python = (self.env_builder)(&recipe, &catalog, &self.workers_dir)?;
        if let Some(mut old) = state.take() {
            old.unload();
        }
        let argv = (self.argv_builder)(&worker_python);
        let root = ollada_root_for_catalog(&self.catalog_dir);
        let mut adapter = WorkerAdapter::new(recipe, &worker_python, &root);
        // The argv builder may have replaced the spawn command (the
        // fake-worker seam); honour it.
        adapter.argv = argv;
        adapter.ready_timeout = self.ready_timeout;
        adapter.kill_timeout = self.kill_timeout;
        let load_result = adapter.load()?;
        if let Some(peak) = load_result.peak_gb {
            record_peak(
                adapter.recipe_name(),
                Some(peak),
                load_result.hardware.as_deref(),
                &self.meas_dir,
            );
        }
        let info = LoadInfo {
            name: adapter.recipe_name().to_string(),
            family: adapter.recipe_family().to_string(),
            supports_watch: load_result.supports_watch,
            peak_gb: load_result.peak_gb,
            hardware: load_result.hardware.clone(),
        };
        *state = Some(adapter);
        Ok(info)
    }

    /// Generate one reply, loading `name` first if needed. One GPU,
    /// one generation at a time: the whole load-plus-chat holds the
    /// manager lock.
    pub fn chat(
        &self,
        name: &str,
        messages: &[ChatMessage],
        overrides: Option<JsonValue>,
        force: bool,
        on_step: Option<StepHandler>,
    ) -> ManagerResult<GenResult> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ManagerError::Worker(WorkerError("manager lock poisoned".to_string())))?;
        let adapter = {
            // ensure_loaded_locked borrows state; call it first, then
            // take the adapter out of the slot for the chat.
            self.ensure_loaded_locked(&mut state, name, force)?;
            state.as_mut().expect("ensure_loaded left an adapter")
        };
        // Watch mode: the step callback reaches only adapters that
        // support it — the same gate as before the worker rewrite
        // (the worker enforces it again on its side, belt and braces).
        let result = if on_step.is_some() && adapter.supports_watch() {
            adapter.chat(messages, overrides, on_step)
        } else {
            adapter.chat(messages, overrides, None)
        };
        let result = result?;
        if let Some(peak) = result.peak_gb {
            record_peak(name, Some(peak), result.hardware.as_deref(), &self.meas_dir);
        }
        Ok(result)
    }

    /// Kill the worker (the clean-VRAM-unload half) and clear state.
    pub fn unload(&self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(mut adapter) = state.take() {
                adapter.unload();
            }
        }
    }
}

impl Drop for ModelManager {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(mut adapter) = state.take() {
                adapter.unload();
            }
        }
    }
}

/// Whether the process is still alive — test helper for the kill
/// tests (sysinfo, cross-platform).
#[cfg(test)]
pub(crate) fn process_alive(pid: u32) -> bool {
    let mut sys = match sysinfo::System::new() {
        Ok(sys) => sys,
        Err(_) => return false,
    };
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    sys.process(sysinfo::Pid::from_u32(pid)).is_some()
}

/// The fake-worker tests share one process with the global live
/// registry (`kill_all_live` kills every registered worker), so
/// they serialise on this lock: a test that kills all live
/// workers must never run while another test's worker is up.
/// Module-level (not inside `mod tests`) so the server tests,
/// which also spawn fake workers, can share it.
#[cfg(test)]
pub(crate) static WORKER_TEST_LOCK: std::sync::LazyLock<Mutex<()>> =
    std::sync::LazyLock::new(|| Mutex::new(()));

#[cfg(test)]
pub(crate) fn worker_test_lock() -> std::sync::MutexGuard<'static, ()> {
    WORKER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::load_catalog as load_cat;

    fn catalog_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ollada")
            .join("catalog")
    }

    fn catalog() -> Vec<Recipe> {
        load_cat(&catalog_dir()).unwrap()
    }

    fn recipe(name: &str) -> Recipe {
        catalog().into_iter().find(|r| r.name == name).unwrap()
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ollada-rs-manager-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn system_python() -> PathBuf {
        crate::workers::system_python().unwrap()
    }

    /// Generous hardware: no GPU (verdict unknown, never blocks) and
    /// plenty of host RAM (no floor block).
    fn generous_hw() -> Hardware {
        Hardware {
            gpu_name: None,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: Some(64.0),
        }
    }

    /// A 1 GB card: sdar:1.7b's 4.1 GB estimate will not fit.
    fn tiny_gpu_hw() -> Hardware {
        Hardware {
            gpu_name: Some("Tiny GPU".to_string()),
            vram_total_gb: Some(1.0),
            vram_free_gb: Some(1.0),
            ram_total_gb: Some(64.0),
        }
    }

    /// A fake worker speaking the protocol. Behaviour is driven by
    /// argv[1] (mode) and argv[2] (pidfile, where it records its pid
    /// so tests can check it is dead after a kill).
    const FAKE_WORKER: &str = r#"
import json, os, sys

mode = sys.argv[1]
pidfile = sys.argv[2]
with open(pidfile, "w") as fh:
    fh.write(str(os.getpid()))

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

emit({"event": "ready", "pid": os.getpid()})
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError:
        continue
    rid = req.get("id")
    op = req.get("op")
    if op == "load":
        if mode == "load-error":
            emit({"id": rid, "ok": False, "error": "load boom",
                  "traceback": "Traceback (most recent call last):\n  load boom"})
            continue
        emit({"id": rid, "ok": True, "result": {
            "loaded": req["model"], "family": "sdar",
            "supports_watch": True, "peak_gb": 1.5, "hardware": "Fake GPU"}})
    elif op == "chat":
        if mode == "die":
            sys.stdout.flush()
            os._exit(3)
        if mode == "error":
            emit({"id": rid, "ok": False, "error": "boom",
                  "traceback": "Traceback (most recent call last):\n  boom"})
            continue
        if mode == "wrong-id":
            emit({"id": 999, "ok": True, "result": {
                "text": "stale", "tokens": 0, "seconds": 0.0}})
        if mode == "steps":
            emit({"event": "step", "pieces": ["hel", "lo"],
                  "block_i": 0, "num_blocks": 2, "step_i": 1, "steps": 4})
            emit({"event": "step", "pieces": ["hello", "world"],
                  "block_i": 1, "num_blocks": 2, "step_i": 2, "steps": 4})
        peak = 2.5 if mode == "peak-chat" else None
        emit({"id": rid, "ok": True, "result": {
            "text": "hello from fake", "tokens": 3, "seconds": 0.5,
            "nfe": None, "peak_gb": peak, "hardware": "Fake GPU"}})
    elif op == "status":
        emit({"id": rid, "ok": True, "result": {"loaded": "sdar:1.7b", "family": "sdar"}})
    else:
        emit({"id": rid, "ok": False, "error": "unknown op %r" % (op,)})
"#;

    fn write_fake(tag: &str) -> (PathBuf, PathBuf) {
        let dir = tmp_dir(&format!("fake-{tag}"));
        let script = dir.join("fake_worker.py");
        std::fs::write(&script, FAKE_WORKER).unwrap();
        (script, dir.join("pid.txt"))
    }

    fn fake_argv(script: &Path, pidfile: &Path, mode: &str) -> Vec<String> {
        vec![
            system_python().to_string_lossy().into_owned(),
            script.to_string_lossy().into_owned(),
            mode.to_string(),
            pidfile.to_string_lossy().into_owned(),
        ]
    }

    fn manager_with_fake(tag: &str, mode: &str) -> (ModelManager, PathBuf, PathBuf) {
        let (script, pidfile) = write_fake(tag);
        let meas = tmp_dir(&format!("meas-{tag}"));
        let workers_dir = tmp_dir(&format!("workers-{tag}"));
        let mut manager = ModelManager::new(&catalog_dir(), &workers_dir, &meas);
        manager.set_hardware_override(generous_hw());
        manager.set_timeouts(Duration::from_secs(10), Duration::from_secs(2));
        let argv = fake_argv(&script, &pidfile, mode);
        manager.set_argv_builder(Arc::new(move |_| argv.clone()));
        // The env builder is a no-op: the fake worker needs no venv.
        manager.set_env_builder(Arc::new(|_, _, _| Ok(PathBuf::from("/nonexistent/python"))));
        (manager, pidfile, meas)
    }

    fn read_pid(pidfile: &Path) -> u32 {
        std::fs::read_to_string(pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    fn user_msg(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::user(text)]
    }

    type StepFrames = Arc<Mutex<Vec<(Option<JsonValue>, Option<JsonValue>)>>>;

    // -- WorkerAdapter: handshake and chat -----------------------------------

    #[test]
    fn ready_handshake_load_and_chat_roundtrip() {
        let _guard = worker_test_lock();
        let (manager, pidfile, _meas) = manager_with_fake("roundtrip", "normal");
        let info = manager.ensure_loaded("sdar:1.7b", false).unwrap();
        assert_eq!(info.name, "sdar:1.7b");
        assert_eq!(info.family, "sdar");
        assert!(info.supports_watch);
        assert_eq!(info.peak_gb, Some(1.5));
        assert_eq!(manager.status().loaded.as_deref(), Some("sdar:1.7b"));
        assert_eq!(manager.status().family.as_deref(), Some("sdar"));
        let pid = read_pid(&pidfile);
        assert!(process_alive(pid));
        let result = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap();
        assert_eq!(result.text, "hello from fake");
        assert_eq!(result.tokens, 3);
        assert_eq!(result.seconds, 0.5);
        assert_eq!(result.nfe, None);
        manager.unload();
        assert_eq!(manager.status().loaded, None);
        assert!(!process_alive(pid), "worker survived unload");
    }

    #[test]
    fn step_events_are_forwarded_in_order() {
        let _guard = worker_test_lock();
        let (manager, _pid, _meas) = manager_with_fake("steps", "steps");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let seen: StepFrames = Arc::new(Mutex::new(Vec::new()));
        let seen2 = Arc::clone(&seen);
        let handler: StepHandler = Box::new(move |_pieces, block_i, _num, step_i, _steps| {
            seen2.lock().unwrap().push((block_i, step_i));
        });
        let result = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, Some(handler))
            .unwrap();
        assert_eq!(result.text, "hello from fake");
        let frames = seen.lock().unwrap().clone();
        assert_eq!(
            frames,
            vec![
                (Some(JsonValue::from(0)), Some(JsonValue::from(1))),
                (Some(JsonValue::from(1)), Some(JsonValue::from(2))),
            ]
        );
    }

    #[test]
    fn mismatched_id_reply_is_ignored() {
        let _guard = worker_test_lock();
        let (manager, _pid, _meas) = manager_with_fake("wrong-id", "wrong-id");
        let result = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap();
        // The stale id-999 reply ("stale") is skipped; the real reply wins.
        assert_eq!(result.text, "hello from fake");
    }

    #[test]
    fn worker_error_appends_traceback() {
        let _guard = worker_test_lock();
        let (manager, _pid, _meas) = manager_with_fake("error", "error");
        let err = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap_err();
        match err {
            ManagerError::Adapter(AdapterError(msg)) => {
                assert_eq!(msg, "boom\nTraceback (most recent call last):\n  boom");
            }
            other => panic!("expected AdapterError, got {other:?}"),
        }
        // The adapter is still loaded after a chat error (Python: the
        // worker stays up; only load errors kill it).
        assert_eq!(manager.status().loaded.as_deref(), Some("sdar:1.7b"));
    }

    #[test]
    fn worker_death_mid_request_reports_exit_code() {
        let _guard = worker_test_lock();
        let (manager, pidfile, _meas) = manager_with_fake("die", "die");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let err = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap_err();
        match err {
            ManagerError::Worker(WorkerError(msg)) => {
                assert_eq!(msg, "worker exited mid-request (rc=3)");
            }
            other => panic!("expected WorkerError, got {other:?}"),
        }
        assert!(!process_alive(read_pid(&pidfile)));
    }

    #[test]
    fn load_error_kills_the_worker() {
        let _guard = worker_test_lock();
        let (manager, pidfile, _meas) = manager_with_fake("load-error", "load-error");
        let err = manager.ensure_loaded("sdar:1.7b", false).unwrap_err();
        match err {
            ManagerError::Adapter(AdapterError(msg)) => {
                assert!(msg.starts_with("load boom\nTraceback"), "{msg}");
            }
            other => panic!("expected AdapterError, got {other:?}"),
        }
        assert_eq!(manager.status().loaded, None);
        assert!(!process_alive(read_pid(&pidfile)));
    }

    // -- WorkerAdapter: broken startups ----------------------------------------

    fn startup_script(tag: &str, body: &str) -> PathBuf {
        let dir = tmp_dir(&format!("startup-{tag}"));
        let script = dir.join("worker.py");
        std::fs::write(&script, body).unwrap();
        script
    }

    fn adapter_for_script(tag: &str, script: &Path) -> WorkerAdapter {
        WorkerAdapter::with_argv(
            recipe("sdar:1.7b"),
            vec![
                system_python().to_string_lossy().into_owned(),
                script.to_string_lossy().into_owned(),
                "normal".to_string(),
                tmp_dir(&format!("startup-pid-{tag}"))
                    .join("pid.txt")
                    .to_string_lossy()
                    .into_owned(),
            ],
        )
        .with_timeouts(Duration::from_secs(10), Duration::from_secs(2))
    }

    #[test]
    fn non_json_first_line() {
        let _guard = worker_test_lock();
        let script = startup_script("nonjson", "print('hello', flush=True)\n");
        let mut adapter = adapter_for_script("nonjson", &script);
        let err = adapter.load().unwrap_err();
        match err {
            ManagerError::Worker(WorkerError(msg)) => {
                assert_eq!(msg, "worker printed a non-JSON first line: 'hello'");
            }
            other => panic!("expected WorkerError, got {other:?}"),
        }
    }

    #[test]
    fn worker_exits_during_startup() {
        let _guard = worker_test_lock();
        let script = startup_script("exits", "import sys; sys.exit(7)\n");
        let mut adapter = adapter_for_script("exits", &script);
        let err = adapter.load().unwrap_err();
        match err {
            ManagerError::Worker(WorkerError(msg)) => {
                assert_eq!(msg, "worker exited during startup (rc=7)");
            }
            other => panic!("expected WorkerError, got {other:?}"),
        }
    }

    #[test]
    fn first_line_not_a_ready_event() {
        let _guard = worker_test_lock();
        let script = startup_script(
            "notready",
            "import json; print(json.dumps({'event': 'nope'}), flush=True)\n",
        );
        let mut adapter = adapter_for_script("notready", &script);
        let err = adapter.load().unwrap_err();
        match err {
            ManagerError::Worker(WorkerError(msg)) => {
                assert_eq!(
                    msg,
                    "worker first line was not a ready event: {'event': 'nope'}"
                );
            }
            other => panic!("expected WorkerError, got {other:?}"),
        }
    }

    #[test]
    fn chat_before_load_is_an_adapter_error() {
        let _guard = worker_test_lock();
        let (script, pidfile) = write_fake("chat-before-load");
        let mut adapter =
            WorkerAdapter::with_argv(recipe("sdar:1.7b"), fake_argv(&script, &pidfile, "normal"));
        let err = adapter.chat(&user_msg("hi"), None, None).unwrap_err();
        match err {
            ManagerError::Adapter(AdapterError(msg)) => {
                assert_eq!(msg, "chat before load");
            }
            other => panic!("expected AdapterError, got {other:?}"),
        }
    }

    // -- ModelManager: ordering guarantees --------------------------------------

    #[test]
    fn fit_refusal_leaves_state_intact() {
        let _guard = worker_test_lock();
        let (manager, pidfile, _meas) = manager_with_fake("fit-refusal", "normal");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let pid = read_pid(&pidfile);
        assert!(process_alive(pid));
        // Now the fit check refuses: same manager, tiny GPU.
        manager.set_hardware_override(tiny_gpu_hw());
        let err = manager.ensure_loaded("sdar:4b", false).unwrap_err();
        match err {
            ManagerError::Fit(FitError(msg)) => {
                assert!(msg.contains("Refusing to load"), "{msg}");
                assert!(msg.contains("--force"), "{msg}");
            }
            other => panic!("expected FitError, got {other:?}"),
        }
        // The resident model is untouched and still answers.
        assert_eq!(manager.status().loaded.as_deref(), Some("sdar:1.7b"));
        assert!(process_alive(pid));
        let result = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap();
        assert_eq!(result.text, "hello from fake");
    }

    #[test]
    fn force_skips_the_fit_check() {
        let _guard = worker_test_lock();
        let (manager, _pid, _meas) = manager_with_fake("force", "normal");
        manager.set_hardware_override(tiny_gpu_hw());
        let info = manager.ensure_loaded("sdar:1.7b", true).unwrap();
        assert_eq!(info.name, "sdar:1.7b");
    }

    #[test]
    fn env_build_failure_keeps_the_loaded_model() {
        let _guard = worker_test_lock();
        let (mut manager, pidfile, _meas) = manager_with_fake("env-fail", "normal");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let pid = read_pid(&pidfile);
        // From here on, every env build fails.
        manager.set_env_builder(Arc::new(|_, _, _| {
            Err(WorkerEnvError("boom: no env".to_string()))
        }));
        let err = manager.ensure_loaded("sdar:4b", false).unwrap_err();
        match err {
            ManagerError::Env(WorkerEnvError(msg)) => {
                assert_eq!(msg, "boom: no env");
            }
            other => panic!("expected WorkerEnvError, got {other:?}"),
        }
        assert_eq!(manager.status().loaded.as_deref(), Some("sdar:1.7b"));
        assert!(process_alive(pid));
        let result = manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap();
        assert_eq!(result.text, "hello from fake");
    }

    #[test]
    fn same_model_returns_the_loaded_adapter_without_rebuilding() {
        let _guard = worker_test_lock();
        let (mut manager, _pid, _meas) = manager_with_fake("same-model", "normal");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        // The env builder now counts calls and would fail if called:
        // the early return must not reach it.
        let calls = Arc::new(Mutex::new(0usize));
        let calls2 = Arc::clone(&calls);
        manager.set_env_builder(Arc::new(move |_, _, _| {
            *calls2.lock().unwrap() += 1;
            Err(WorkerEnvError("must not be called".to_string()))
        }));
        let info = manager.ensure_loaded("sdar:1.7b", false).unwrap();
        assert_eq!(info.name, "sdar:1.7b");
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn inprocess_is_refused_and_leaves_state_intact() {
        let _guard = worker_test_lock();
        let (manager, pidfile, _meas) = manager_with_fake("inprocess", "normal");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let pid = read_pid(&pidfile);
        manager.set_inprocess_override(true);
        let err = manager.ensure_loaded("sdar:4b", false).unwrap_err();
        match err {
            ManagerError::InProcess(msg) => {
                assert!(
                    msg.contains("in-process loading is not supported by the Rust manager"),
                    "{msg}"
                );
            }
            other => panic!("expected InProcess refusal, got {other:?}"),
        }
        assert_eq!(manager.status().loaded.as_deref(), Some("sdar:1.7b"));
        assert!(process_alive(pid));
    }

    #[test]
    fn inprocess_refusal_with_nothing_loaded() {
        let _guard = worker_test_lock();
        let (manager, _pid, _meas) = manager_with_fake("inprocess-empty", "normal");
        manager.set_inprocess_override(true);
        let err = manager.ensure_loaded("sdar:1.7b", false).unwrap_err();
        assert!(matches!(err, ManagerError::InProcess(_)));
        assert_eq!(manager.status().loaded, None);
    }

    // -- measurements ------------------------------------------------------------

    #[test]
    fn load_and_chat_peaks_are_recorded() {
        let _guard = worker_test_lock();
        let (manager, _pid, meas) = manager_with_fake("peaks", "peak-chat");
        manager.ensure_loaded("sdar:1.7b", false).unwrap();
        let after_load = load_record("sdar:1.7b", &meas).unwrap();
        assert_eq!(after_load.peak_vram_gb, Some(1.5));
        assert_eq!(after_load.observations, Some(1));
        assert_eq!(after_load.hardware.as_deref(), Some("Fake GPU"));
        manager
            .chat("sdar:1.7b", &user_msg("hi"), None, false, None)
            .unwrap();
        let after_chat = load_record("sdar:1.7b", &meas).unwrap();
        assert_eq!(after_chat.peak_vram_gb, Some(2.5)); // max wins
        assert_eq!(after_chat.observations, Some(2));
    }

    // -- kill-all ------------------------------------------------------------------

    #[test]
    fn kill_all_live_terminates_registered_workers() {
        let _guard = worker_test_lock();
        let (script, pidfile) = write_fake("kill-all");
        let pidfile2 = tmp_dir("kill-all-2").join("pid.txt");
        let mut a =
            WorkerAdapter::with_argv(recipe("sdar:1.7b"), fake_argv(&script, &pidfile, "normal"))
                .with_timeouts(Duration::from_secs(10), Duration::from_secs(2));
        let mut b =
            WorkerAdapter::with_argv(recipe("sdar:4b"), fake_argv(&script, &pidfile2, "normal"))
                .with_timeouts(Duration::from_secs(10), Duration::from_secs(2));
        a.load().unwrap();
        b.load().unwrap();
        let pid_a = read_pid(&pidfile);
        let pid_b = read_pid(&pidfile2);
        assert!(process_alive(pid_a) && process_alive(pid_b));
        kill_all_live();
        // The adapters notice the death on next use (proc reaped) —
        // reap first: until then the workers are zombies, which
        // still show in the process table.
        assert!(!a.loaded());
        assert!(!b.loaded());
        assert!(!process_alive(pid_a), "worker A survived kill_all_live");
        assert!(!process_alive(pid_b), "worker B survived kill_all_live");
    }

    #[test]
    fn ollada_root_for_catalog_finds_the_repo_root() {
        let root = ollada_root_for_catalog(&catalog_dir());
        assert_eq!(root, PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."));
        assert!(root.join("ollada").join("worker.py").is_file());
    }
}
