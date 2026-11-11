//! The Phase 0 commands: `list`, `show`, `recommend`, `ps`, `rm`.
//!
//! Each function mirrors its Python counterpart in `ollada/cli.py`
//! line for line, including column widths, float formatting, and the
//! exact wording of the fit notes. They write to a `Write` sink and
//! take their inputs (catalog dir, measurements dir, hardware) as
//! parameters, so the parity tests can run them against the real
//! catalog with fabricated hardware and compare byte-for-byte with
//! the Python CLI's output.
//!
//! Deliberate differences from Python, all documented:
//! - `show`'s `measured here` line: Python crashes with KeyError if a
//!   record lacks `observations`; this crate prints 1 instead. Real
//!   records always carry it.
//! - Hardware detection has no torch path (see fit.rs).

use crate::cache::{find_cached_repo, format_size, hf_cache_dir};
use crate::fit::{assess, detect_hardware, Hardware};
use crate::http::get_json;
use crate::manager::{ChatMessage, ManagerError, ModelManager};
use crate::measurements::{load_record, measurements_dir};
use crate::pyrepr::{py_float, py_json, py_yaml};
use crate::recipes::{get_recipe, load_catalog, Recipe, RecipeError};
use crate::workers::{list_envs, rebuild_env, Runner, WorkerEnvError};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Error type for commands: mirrors Python's `main()` handling, where
/// a RecipeError prints `error: ...` to stderr and exits 2.
#[derive(Debug)]
pub struct CmdError(pub String);

impl From<RecipeError> for CmdError {
    fn from(e: RecipeError) -> Self {
        CmdError(e.0)
    }
}

impl From<WorkerEnvError> for CmdError {
    fn from(e: WorkerEnvError) -> Self {
        CmdError(e.0)
    }
}

pub type CmdResult<T> = Result<T, CmdError>;

fn write_line(out: &mut dyn Write, line: &str) {
    let _ = writeln!(out, "{line}");
}

/// `ollada list` — mirrors `_cmd_list`.
pub fn cmd_list(catalog_dir: &Path, out: &mut dyn Write) -> CmdResult<i32> {
    let recipes = load_catalog(catalog_dir)?;
    if recipes.is_empty() {
        write_line(out, "the shelf is empty");
        return Ok(0);
    }
    write_line(
        out,
        &format!(
            "{:<16} {:<8} {:<11} {:<6} {}",
            "NAME", "TASK", "SIZE(bf16)", "QUANT", "VERIFIED"
        ),
    );
    for r in &recipes {
        write_line(
            out,
            &format!(
                "{:<16} {:<8} {:<11} {:<6} {}",
                r.name,
                r.task,
                py_float(r.size_gb_bf16),
                r.quant_default,
                r.verified_hardware(),
            ),
        );
    }
    Ok(0)
}

/// `ollada show <name>` — mirrors `_cmd_show`.
/// `hardware` is injected (main detects it live; tests fabricate it).
pub fn cmd_show(
    catalog_dir: &Path,
    meas_dir: &Path,
    name: &str,
    hardware: &Hardware,
    out: &mut dyn Write,
) -> CmdResult<i32> {
    let r = get_recipe(name, catalog_dir)?;
    write_line(out, &format!("name:        {}", r.name));
    write_line(out, &format!("family:      {}", r.family));
    write_line(out, &format!("source:      {}", r.source));
    write_line(out, &format!("license:     {}", r.license));
    write_line(out, &format!("task:        {}", r.task));
    write_line(out, &format!("description: {}", r.description));
    write_line(
        out,
        &format!("size (bf16): {} GB", py_float(r.size_gb_bf16)),
    );
    write_line(
        out,
        &format!("quant:       {}   dtype: {}", r.quant_default, r.dtype),
    );
    let pin = match r.env.get("transformers") {
        Some(serde_yaml::Value::String(s)) if !s.is_empty() => s.clone(),
        _ => " (pin set at verification)".to_string(),
    };
    write_line(out, &format!("env:         transformers{pin}"));
    write_line(out, &format!("generate:    {}", py_yaml(&r.generate)));
    if r.is_verified() {
        write_line(out, &format!("verified:    {}", py_yaml(&r.verified)));
    } else {
        write_line(
            out,
            "verified:    no — defaults are provisional until a real-load run (PLAN.md §7)",
        );
    }
    if let Some(record) = load_record(&r.name, meas_dir) {
        let n = record.observations.unwrap_or(1);
        let plural = if n != 1 { "s" } else { "" };
        write_line(
            out,
            &format!(
                "measured here: peak {:.2} GB on {} ({} observation{}, updated {})",
                record.peak_vram_gb.unwrap_or(0.0),
                record
                    .hardware
                    .clone()
                    .unwrap_or_else(|| "this machine".to_string()),
                n,
                plural,
                record.updated.clone().unwrap_or_default(),
            ),
        );
    }
    let report = assess(&r, hardware, load_record(&r.name, meas_dir).as_ref());
    write_line(out, &format!("fit here:    {}", report.summary()));
    for note in &report.notes {
        write_line(out, &format!("             note: {note}"));
    }
    Ok(0)
}

/// `ollada recommend` — mirrors `_cmd_recommend`.
pub fn cmd_recommend(
    catalog_dir: &Path,
    meas_dir: &Path,
    hardware: &Hardware,
    out: &mut dyn Write,
) -> CmdResult<i32> {
    if let Some(gpu) = &hardware.gpu_name {
        let vram = match hardware.vram_total_gb {
            Some(total) => format!("{total:.1} GB VRAM"),
            None => "VRAM unknown".to_string(),
        };
        write_line(out, &format!("this machine: {gpu} — {vram}"));
    } else {
        write_line(
            out,
            "this machine: no GPU detected — verdicts unavailable, peaks shown for reference",
        );
    }
    if let Some(ram) = hardware.ram_total_gb {
        write_line(out, &format!("host RAM: {ram:.1} GB"));
    }
    write_line(out, "");
    write_line(
        out,
        &format!(
            "{:<16} {:<10} {:<9} {}",
            "NAME", "VERDICT", "PEAK GB", "SOURCE"
        ),
    );
    for r in load_catalog(catalog_dir)? {
        let report = assess(&r, hardware, load_record(&r.name, meas_dir).as_ref());
        let peak = match report.peak_gb {
            Some(p) => format!("{p:.2}"),
            None => "—".to_string(),
        };
        write_line(
            out,
            &format!(
                "{:<16} {:<10} {:<9} {}",
                r.name, report.verdict, peak, report.peak_source
            ),
        );
        for note in &report.notes {
            // Python: f"{'':<16} note: {note}" — 16 padded spaces plus
            // the literal space before "note:", 17 in total.
            write_line(out, &format!("                 note: {note}"));
        }
    }
    Ok(0)
}

/// `ollada ps` — mirrors `_cmd_ps`: ask the daemon what is loaded.
/// Any failure (no daemon, non-200, bad JSON) prints the same line
/// Python prints and exits 1.
pub fn cmd_ps(port: u16, out: &mut dyn Write) -> i32 {
    match get_json(port, "/api/ps") {
        Ok(value) => {
            write_line(out, &py_json(&value));
            0
        }
        Err(_) => {
            write_line(
                out,
                "no ollada daemon answering — is `ollada serve` running?",
            );
            1
        }
    }
}

/// `ollada rm <name>` — mirrors `_cmd_rm`: locate a model's cached weights.
pub fn cmd_rm(catalog_dir: &Path, name: &str, out: &mut dyn Write) -> CmdResult<i32> {
    let r: Recipe = get_recipe(name, catalog_dir)?;
    let cache = hf_cache_dir();
    match find_cached_repo(&r.source, &cache) {
        Some(size) => {
            write_line(
                out,
                &format!("found cached copy of {} ({})", r.source, format_size(size)),
            );
            write_line(
                out,
                "delete it with:  huggingface-cli delete-cache  (interactive, safe)",
            );
        }
        None => {
            write_line(out, &format!("{} is not in the local cache", r.source));
        }
    }
    Ok(0)
}

/// `ollada workers` / `ollada workers list` — mirrors `_cmd_workers`
/// with no subcommand: one row per family.
pub fn cmd_workers_list(
    catalog_dir: &Path,
    workers_dir: &Path,
    runner: &dyn Runner,
    out: &mut dyn Write,
) -> CmdResult<i32> {
    let recipes = load_catalog(catalog_dir)?;
    write_line(
        out,
        &format!(
            "{:<12} {:<8} {:<14} {}",
            "FAMILY", "BUILT", "TRANSFORMERS", "PATH"
        ),
    );
    for row in list_envs(&recipes, workers_dir, runner) {
        write_line(
            out,
            &format!(
                "{:<12} {:<8} {:<14} {}",
                row.family,
                if row.built { "yes" } else { "no" },
                row.transformers.as_deref().unwrap_or("—"),
                row.path.display(),
            ),
        );
    }
    Ok(0)
}

/// `ollada workers rebuild <name>` — mirrors `_cmd_workers` rebuild:
/// delete the family's env and rebuild it, then print the confirmation.
/// `hardware`, when given, is used instead of a live detection (the
/// tests inject it; the CLI passes None).
pub fn cmd_workers_rebuild(
    catalog_dir: &Path,
    name: &str,
    workers_dir: &Path,
    runner: &dyn Runner,
    hardware: Option<Hardware>,
    out: &mut dyn Write,
) -> CmdResult<i32> {
    let recipe = get_recipe(name, catalog_dir)?;
    let catalog = load_catalog(catalog_dir)?;
    rebuild_env(&recipe, &catalog, workers_dir, runner, hardware)?;
    write_line(out, &format!("rebuilt worker env for {}", recipe.family));
    Ok(0)
}

/// `ollada run <name> [prompt]` — mirrors `_cmd_run` (one-shot with
/// a prompt, REPL loop without one).
///
/// Deliberate differences from Python, all documented:
/// - **No watch / spinner.** Python shows a live diffusion canvas on
///   a tty (`_CanvasWatch`) and a `working… Ns` spinner on stderr.
///   This prints only the answer on stdout and the `({elapsed:.1f}s)`
///   footer on stderr — the same two lines a piped Python run
///   produces. The step protocol the watch needs is in the manager
///   (`StepHandler`); a renderer is future work, not a Phase 3 gap
///   in the manager itself.
/// - **Ctrl-C kills the process.** Python catches `KeyboardInterrupt`
///   in the REPL and exits 0; Rust has no signal handler here, so
///   SIGINT terminates the process (the atexit hook still kills the
///   worker). The one-shot path is unaffected.
/// - **Worker/adapter errors print one line and exit 1.** Python
///   lets them traceback (exit 1). The message is the same text.
/// - `int()` accepts `1_0` and `+5` in Python's `/length` parser;
///   Rust's parser accepts only plain digits (and a leading `-`,
///   which then fails the positivity check like Python's).
///
/// Options for `cmd_run` — prompt (`None` means the REPL loop),
/// force, and length.
pub struct RunOptions<'a> {
    pub prompt: Option<&'a str>,
    pub force: bool,
    pub length: Option<i64>,
}

pub fn cmd_run(
    manager: &ModelManager,
    name: &str,
    opts: RunOptions<'_>,
    out: &mut dyn Write,
    err: &mut dyn Write,
    input: &mut dyn BufRead,
) -> i32 {
    let RunOptions {
        prompt,
        force,
        length,
    } = opts;
    if let Some(n) = length {
        if n < 1 {
            let _ = writeln!(err, "length must be a positive number of tokens");
            return 2;
        }
    }
    let mut session_length = length;
    if let Some(prompt) = prompt {
        return match run_one_chat(
            manager,
            name,
            &[ChatMessage::user(prompt)],
            session_length,
            force,
            out,
            err,
        ) {
            Ok(_) => 0,
            Err(code) => code,
        };
    }
    let _ = writeln!(
        out,
        "{name} — type a message, Ctrl-C to quit · /length <n> sets the canvas size"
    );
    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        let _ = write!(out, ">>> ");
        let _ = out.flush();
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                let _ = writeln!(out);
                return 0; // EOF
            }
            Ok(_) => {}
            Err(_) => {
                let _ = writeln!(out);
                return 0;
            }
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match parse_length_command(text) {
            Ok(Some(n)) => {
                session_length = Some(n);
                let _ = writeln!(out, "canvas length: {n} tokens");
                continue;
            }
            Ok(None) => {}
            Err(msg) => {
                let _ = writeln!(out, "{msg}");
                continue;
            }
        }
        history.push(ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
        });
        match run_one_chat(manager, name, &history, session_length, force, out, err) {
            Ok(reply) => history.push(ChatMessage {
                role: "assistant".to_string(),
                content: reply,
            }),
            Err(code) => return code,
        }
    }
}

/// Parse the REPL's `/length` command. Mirrors
/// `ollada.cli._parse_length_command`.
fn parse_length_command(text: &str) -> Result<Option<i64>, String> {
    let stripped = text.trim();
    if stripped != "/length" && !stripped.starts_with("/length ") {
        return Ok(None);
    }
    let rest = stripped["/length".len()..].trim();
    match rest.parse::<i64>() {
        Err(_) => Err("usage: /length <positive number of tokens>".to_string()),
        Ok(value) if value < 1 => Err("length must be a positive number of tokens".to_string()),
        Ok(value) => Ok(Some(value)),
    }
}

/// One generation: the answer on stdout, the `({elapsed:.1f}s)`
/// footer on stderr — exactly Python's one-shot shape.
fn run_one_chat(
    manager: &ModelManager,
    name: &str,
    messages: &[ChatMessage],
    length: Option<i64>,
    force: bool,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<String, i32> {
    let overrides = length.map(|n| serde_json::json!({"gen_length": n}));
    let start = Instant::now();
    match manager.chat(name, messages, overrides, force, None) {
        Ok(result) => {
            let _ = writeln!(out, "{}", result.text);
            let _ = writeln!(err, "({:.1}s)", start.elapsed().as_secs_f64());
            Ok(result.text)
        }
        Err(e) => Err(print_manager_error(&e, err)),
    }
}

/// Print a manager error the way Python's `main()` maps it, and
/// return the exit code: fit/refusal errors exit 2, worker and
/// adapter failures exit 1.
fn print_manager_error(e: &ManagerError, err: &mut dyn Write) -> i32 {
    match e {
        ManagerError::Fit(msg) => {
            let _ = writeln!(err, "fit check: {msg}");
            2
        }
        ManagerError::Recipe(msg) => {
            let _ = writeln!(err, "error: {msg}");
            2
        }
        ManagerError::Env(msg) => {
            let _ = writeln!(err, "error: {msg}");
            2
        }
        ManagerError::Worker(msg) => {
            let _ = writeln!(err, "error: {msg}");
            1
        }
        ManagerError::Adapter(msg) => {
            let _ = writeln!(err, "error: {msg}");
            1
        }
        ManagerError::InProcess(msg) => {
            let _ = writeln!(err, "error: {msg}");
            1
        }
    }
}

/// Resolve the catalog directory: `$OLLADA_CATALOG_DIR`, else walk up
/// from the executable looking for `ollada/catalog` (the repo layout),
/// else `./ollada/catalog` from the working directory, else `./catalog`.
pub fn resolve_catalog_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLADA_CATALOG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..5 {
            if let Some(d) = dir {
                let candidate = d.join("ollada").join("catalog");
                if candidate.is_dir() {
                    return candidate;
                }
                dir = d.parent().map(|p| p.to_path_buf());
            } else {
                break;
            }
        }
    }
    let cwd_catalog = PathBuf::from("ollada").join("catalog");
    if cwd_catalog.is_dir() {
        return cwd_catalog;
    }
    PathBuf::from("catalog")
}

/// Detect hardware the way the live commands do.
pub fn live_hardware() -> Hardware {
    detect_hardware()
}

/// The live measurements directory.
pub fn live_measurements_dir() -> PathBuf {
    measurements_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Hardware;

    /// The real catalog, two levels up from this crate.
    fn catalog() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ollada")
            .join("catalog")
    }

    /// A measurements dir with no records (mirrors this sandbox).
    fn empty_meas() -> PathBuf {
        let dir = std::env::temp_dir().join("ollada-rs-test-empty-meas");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// This sandbox's hardware as Python sees it: no GPU, 7.7 GB host RAM.
    /// (Python's sysconf value is 7.679... GB; either rounds to 7.7.)
    fn sandbox_hw() -> Hardware {
        Hardware {
            gpu_name: None,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: Some(7.679),
        }
    }

    fn run_list() -> String {
        let mut buf: Vec<u8> = Vec::new();
        cmd_list(&catalog(), &mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn list_matches_python_output() {
        // Captured from `python -m ollada.cli list` on this machine.
        // (concat! keeps leading spaces literal; backslash-continuation
        // would strip them.)
        let expected = concat!(
            "NAME             TASK     SIZE(bf16)  QUANT  VERIFIED\n",
            "diffcoder:8b     coding   16.0        int8   Tesla T4 (Colab)\n",
            "diffcoder:8b4bit coding   16.0        int4   NVIDIA A100-SXM4-80GB (Colab)\n",
            "dream:7b         chat     15.2        int8   Tesla T4 (Colab)\n",
            "nemotron:3b      chat     7.66        bf16   Laptop — Intel Core Ultra 7 365 (8 cores), 32GB RAM, Windows 11; CPU only, bf16\n",
            "sdar:1.7b        chat     4.1         bf16   Laptop — Intel Core Ultra 7 365 (8 cores), 32GB RAM, Windows 11; CPU only, bf16\n",
            "sdar:4b          chat     8.8         bf16   Tesla T4 (Colab, High-RAM runtime)\n",
        );
        assert_eq!(run_list(), expected);
    }

    #[test]
    fn recommend_matches_python_output() {
        let mut buf: Vec<u8> = Vec::new();
        cmd_recommend(&catalog(), &empty_meas(), &sandbox_hw(), &mut buf).unwrap();
        let got = String::from_utf8(buf).unwrap();
        let expected = concat!(
            "this machine: no GPU detected — verdicts unavailable, peaks shown for reference\n",
            "host RAM: 7.7 GB\n",
            "\n",
            "NAME             VERDICT    PEAK GB   SOURCE\n",
            "diffcoder:8b     unknown    15.15     ollada-verified on Tesla T4 (Colab)\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
            "diffcoder:8b4bit unknown    6.59      ollada-verified on NVIDIA A100-SXM4-80GB (Colab)\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
            "dream:7b         unknown    9.80      ollada-verified on Tesla T4 (Colab)\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
            "nemotron:3b      unknown    7.66      weights-size estimate (unverified)\n",
            "                 note: host RAM 7.7 GB is below this recipe's 16 GB floor\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
            "sdar:1.7b        unknown    4.10      weights-size estimate (unverified)\n",
            "                 note: host RAM 7.7 GB is below this recipe's 8 GB floor\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
            "sdar:4b          unknown    8.87      ollada-verified on Tesla T4 (Colab, High-RAM runtime)\n",
            "                 note: host RAM 7.7 GB is below this recipe's 16 GB floor\n",
            "                 note: no GPU detected — VRAM verdict unavailable\n",
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn show_diffcoder_8b4bit_matches_python_output() {
        let mut buf: Vec<u8> = Vec::new();
        cmd_show(
            &catalog(),
            &empty_meas(),
            "diffcoder:8b4bit",
            &sandbox_hw(),
            &mut buf,
        )
        .unwrap();
        let got = String::from_utf8(buf).unwrap();
        let expected = concat!(
            "name:        diffcoder:8b4bit\n",
            "family:      diffcoder\n",
            "source:      ByteDance-Seed/Stable-DiffCoder-8B-Instruct\n",
            "license:     MIT\n",
            "task:        coding\n",
            "description: DiffCoder 8B in 4-bit (NF4 weight-only) — same coding model, ~7 GB peak, fits 8 GB-class cards with headroom on a 16 GB card.\n",
            "size (bf16): 16.0 GB\n",
            "quant:       int4   dtype: float16\n",
            "env:         transformers==5.3.0\n",
            "generate:    {'steps': 256, 'gen_length': 256, 'block_length': 4, 'remasking': 'low_confidence', 'temperature': 0.0, 'threshold': None}\n",
            "verified:    {'hardware': 'NVIDIA A100-SXM4-80GB (Colab)', 'date': '2026-11-06', 'vram_gb': 6.59, 'notes': 'Verified through ollada run (diffcoder worker, transformers 5.3.0, NF4 weight-only), fixed set 3/3: train 90 km/h (35.4s), capital Paris (18.7s), fib correct (26.2s). Peak VRAM 6.59 GB over 6 observations. Desktop 5060 Ti verification still pending.'}\n",
            "fit here:    diffcoder:8b4bit: unknown — peak 6.59 GB (ollada-verified on NVIDIA A100-SXM4-80GB (Colab)), needs ~7.6 GB with headroom; this GPU has unknown VRAM\n",
            "             note: no GPU detected — VRAM verdict unavailable\n",
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn show_sdar_17b_matches_python_output() {
        let mut buf: Vec<u8> = Vec::new();
        cmd_show(
            &catalog(),
            &empty_meas(),
            "sdar:1.7b",
            &sandbox_hw(),
            &mut buf,
        )
        .unwrap();
        let got = String::from_utf8(buf).unwrap();
        // The verified.notes string contains a single quote, so Python
        // repr switches it to double quotes — the pyrepr test for that
        // is the load-bearing check here.
        assert!(got.starts_with("name:        sdar:1.7b\nfamily:      sdar\n"));
        assert!(got.contains("generate:    {'gen_length': 128, 'block_length': 4, 'denoising_steps': 4, 'remasking': 'low_confidence_dynamic', 'confidence_threshold': 0.9, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}\n"));
        assert!(got.contains("\"Verified on CPU through ollada's own adapter"));
        assert!(got.contains("fit here:    sdar:1.7b: unknown — peak 4.10 GB (weights-size estimate (unverified)), needs ~5.1 GB with headroom; this GPU has unknown VRAM\n"));
        assert!(
            got.contains("             note: host RAM 7.7 GB is below this recipe's 8 GB floor\n")
        );
        assert!(got.contains("             note: no GPU detected — VRAM verdict unavailable\n"));
    }

    #[test]
    fn show_unknown_model_errors_like_python() {
        let mut buf: Vec<u8> = Vec::new();
        let err = cmd_show(
            &catalog(),
            &empty_meas(),
            "nope:1b",
            &sandbox_hw(),
            &mut buf,
        )
        .unwrap_err();
        assert!(err.0.contains("not on the shelf"));
    }

    #[test]
    fn workers_list_matches_python_output() {
        // Captured from `OLLADA_WORKERS_DIR=<dir> python -m ollada.cli
        // workers` on this machine (same shape; the path is the dir).
        let dir = std::env::temp_dir().join("ollada-rs-test-workers-list");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let runner = crate::workers::FakeRunner::ok();
        let mut buf: Vec<u8> = Vec::new();
        let code = cmd_workers_list(&catalog(), &dir, &runner, &mut buf).unwrap();
        assert_eq!(code, 0);
        let d = dir.display();
        let expected = format!(
            "FAMILY       BUILT    TRANSFORMERS   PATH\n\
             diffcoder    no       —              {d}/diffcoder\n\
             dream        no       —              {d}/dream\n\
             nemotron     no       —              {d}/nemotron\n\
             sdar         no       —              {d}/sdar\n"
        );
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn workers_rebuild_reports_the_family() {
        // Fake runner: every subprocess succeeds; CPU hardware, so no
        // CUDA check and no installed-transformers answer (None skips
        // the sibling guard, like Python's `if installed is not None`).
        let dir = std::env::temp_dir().join("ollada-rs-test-workers-rebuild-ok");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let runner = crate::workers::FakeRunner::ok();
        let mut buf: Vec<u8> = Vec::new();
        let code = cmd_workers_rebuild(
            &catalog(),
            "dream:7b",
            &dir,
            &runner,
            Some(Hardware {
                gpu_name: None,
                ram_total_gb: Some(32.0),
                ..Default::default()
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(code, 0);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "rebuilt worker env for dream\n"
        );
    }

    #[test]
    fn workers_rebuild_unknown_model_errors_like_python() {
        let dir = std::env::temp_dir().join("ollada-rs-test-workers-rebuild");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let runner = crate::workers::FakeRunner::ok();
        let mut buf: Vec<u8> = Vec::new();
        let err =
            cmd_workers_rebuild(&catalog(), "nope:1b", &dir, &runner, None, &mut buf).unwrap_err();
        assert!(err.0.contains("not on the shelf"));
    }

    #[test]
    fn ps_with_no_daemon_matches_python() {
        let mut buf: Vec<u8> = Vec::new();
        // Port 1 is unconnectable; Python prints the same line, exit 1.
        let code = cmd_ps(1, &mut buf);
        assert_eq!(code, 1);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "no ollada daemon answering — is `ollada serve` running?\n"
        );
    }

    #[test]
    fn rm_reports_a_cached_model_like_python() {
        let mut buf: Vec<u8> = Vec::new();
        let code = cmd_rm(&catalog(), "diffcoder:8b4bit", &mut buf).unwrap();
        assert_eq!(code, 0);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "found cached copy of ByteDance-Seed/Stable-DiffCoder-8B-Instruct (12.0M)\n\
             delete it with:  huggingface-cli delete-cache  (interactive, safe)\n"
        );
    }

    #[test]
    fn rm_reports_a_missing_model() {
        let mut buf: Vec<u8> = Vec::new();
        let code = cmd_rm(&catalog(), "nemotron:3b", &mut buf).unwrap();
        assert_eq!(code, 0);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "nvidia/Nemotron-Labs-Diffusion-3B is not in the local cache\n"
        );
    }

    #[test]
    fn run_length_validation_exits_2_without_touching_a_model() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let mut input: &[u8] = b"";
        let dir = std::env::temp_dir().join("ollada-rs-test-run-length");
        let manager = ModelManager::new(&catalog(), &dir, &dir);
        let code = cmd_run(
            &manager,
            "sdar:1.7b",
            RunOptions {
                prompt: Some("hi"),
                force: false,
                length: Some(0),
            },
            &mut out,
            &mut err,
            &mut input,
        );
        assert_eq!(code, 2);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "length must be a positive number of tokens\n"
        );
        assert!(out.is_empty());
    }

    #[test]
    fn run_unknown_model_errors_like_python() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let mut input: &[u8] = b"";
        let dir = std::env::temp_dir().join("ollada-rs-test-run-unknown");
        let manager = ModelManager::new(&catalog(), &dir, &dir);
        let code = cmd_run(
            &manager,
            "nope:1b",
            RunOptions {
                prompt: Some("hi"),
                force: false,
                length: None,
            },
            &mut out,
            &mut err,
            &mut input,
        );
        assert_eq!(code, 2);
        let stderr = String::from_utf8(err).unwrap();
        assert!(stderr.contains("not on the shelf"), "stderr: {stderr}");
    }

    #[test]
    fn parse_length_command_matches_python() {
        assert_eq!(parse_length_command("hello").unwrap(), None);
        assert_eq!(parse_length_command("/length 128").unwrap(), Some(128));
        assert_eq!(parse_length_command("  /length 64  ").unwrap(), Some(64));
        assert_eq!(
            parse_length_command("/length").unwrap_err(),
            "usage: /length <positive number of tokens>"
        );
        assert_eq!(
            parse_length_command("/length abc").unwrap_err(),
            "usage: /length <positive number of tokens>"
        );
        assert_eq!(
            parse_length_command("/length 0").unwrap_err(),
            "length must be a positive number of tokens"
        );
        assert_eq!(
            parse_length_command("/length -3").unwrap_err(),
            "length must be a positive number of tokens"
        );
        // "/lengthwise" is not the command.
        assert_eq!(parse_length_command("/lengthwise 5").unwrap(), None);
    }

    #[test]
    fn run_repl_quits_on_eof() {
        // No prompt: the REPL banner prints, then EOF ends it with 0.
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let mut input: &[u8] = b"";
        let dir = std::env::temp_dir().join("ollada-rs-test-run-repl");
        let manager = ModelManager::new(&catalog(), &dir, &dir);
        let code = cmd_run(
            &manager,
            "sdar:1.7b",
            RunOptions {
                prompt: None,
                force: false,
                length: None,
            },
            &mut out,
            &mut err,
            &mut input,
        );
        assert_eq!(code, 0);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "sdar:1.7b — type a message, Ctrl-C to quit · /length <n> sets the canvas size\n>>> \n"
        );
    }
}
