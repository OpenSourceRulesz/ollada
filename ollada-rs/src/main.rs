//! ollada-rs — the ollada Rust front-end.
//!
//! Read-only shell (`list`, `show`, `recommend`, `ps`, `rm`, plus the
//! fit check), worker environment management (`workers`,
//! `workers rebuild`), the model manager (`run`) speaking the
//! JSON-lines worker protocol, and the HTTP server (`serve`, an axum
//! port of `server.py`). No model code moves, nothing the user
//! depends on changes. The Python CLI stays the reference
//! implementation; output must match it (see the parity tests in
//! `commands.rs` and `server.rs`).
//!
//! Usage differences from the Python CLI, all deliberate:
//! - the binary is named `ollada` (it IS the `ollada` command when
//!   installed from the wheel); `--version` prints `ollada 0.3.7`;
//! - `pull` shells out to `python -m ollada.cli pull` (see
//!   `commands::cmd_pull`): downloading weights is the Python
//!   `huggingface-hub` path, and this crate has none of its own;
//! - hardware detection has no torch path (see `fit.rs`);
//! - the catalog is found via `$OLLADA_CATALOG_DIR`, or by walking up
//!   from the executable to `ollada/catalog` (the repo layout), or
//!   `./ollada/catalog` from the working directory.
//! - venvs are built with the system Python found on `PATH`
//!   (`$OLLADA_PYTHON` overrides); see `workers.rs`.
//! - `run` has no live canvas watch or spinner (see `commands.rs`);
//!   the answer and the `({elapsed:.1f}s)` footer match Python's
//!   piped output exactly.
//! - `serve` reports the ollada version (0.3.7, from
//!   `ollada/__init__.py`) on its API, not the crate version.

mod cache;
mod commands;
mod fit;
mod http;
mod manager;
mod measurements;
mod pyrepr;
mod recipes;
mod server;
mod workers;

use clap::{CommandFactory, Parser, Subcommand};
use std::io::Write;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "ollada",
    version,
    about = "Ollada Rust front-end (read-only shell + worker envs + model manager + HTTP server). The shelf for open-source diffusion language models — pull, run, and serve dLLMs locally, Ollama-style."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Show the shelf.
    List,
    /// Show one model's recipe.
    Show { name: String },
    /// Download a model's weights (via the Python CLI — see cmd_pull).
    Pull { name: String },
    /// Chat with a model (one-shot with a prompt arg).
    Run {
        name: String,
        prompt: Option<String>,
        /// Load even if the fit check refuses.
        #[arg(long)]
        force: bool,
        /// Canvas size in tokens (default: the recipe's gen_length).
        #[arg(long)]
        length: Option<i64>,
    },
    /// Which shelf models fit this machine.
    Recommend,
    /// Run the local API daemon.
    Serve {
        /// Interface to bind (localhost-only by default).
        #[arg(long, default_value = server::DEFAULT_HOST)]
        host: String,
        /// Port to bind.
        #[arg(long, default_value_t = server::DEFAULT_PORT)]
        port: u16,
    },
    /// Ask the daemon what is loaded.
    Ps {
        /// Daemon port.
        #[arg(long, default_value_t = 11435)]
        port: u16,
    },
    /// Locate a model's cached weights.
    Rm { name: String },
    /// Inspect or rebuild per-family worker envs.
    Workers {
        #[command(subcommand)]
        command: Option<WorkersCommand>,
    },
}

#[derive(Subcommand)]
enum WorkersCommand {
    /// Show worker envs and their transformers versions.
    List,
    /// Delete and rebuild one family's worker env.
    Rebuild {
        /// A model name, e.g. dream:7b.
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let Some(command) = cli.command else {
        // No subcommand: print help, exit 0 — like Python's argparse.
        let _ = Cli::command().print_help();
        println!();
        return;
    };
    let catalog_dir = commands::resolve_catalog_dir();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let code: i32 = match command {
        Command::List => match commands::cmd_list(&catalog_dir, &mut out) {
            Ok(code) => code,
            Err(e) => fail(&e.0),
        },
        Command::Show { name } => {
            let hw = commands::live_hardware();
            let meas = commands::live_measurements_dir();
            match commands::cmd_show(&catalog_dir, &meas, &name, &hw, &mut out) {
                Ok(code) => code,
                Err(e) => fail(&e.0),
            }
        }
        Command::Pull { name } => commands::cmd_pull(&name),
        Command::Run {
            name,
            prompt,
            force,
            length,
        } => {
            let stderr = std::io::stderr();
            let mut err = stderr.lock();
            let stdin = std::io::stdin();
            let mut input = stdin.lock();
            let manager = manager::ModelManager::new(
                &catalog_dir,
                &workers::workers_dir(),
                &commands::live_measurements_dir(),
            );
            commands::cmd_run(
                &manager,
                &name,
                commands::RunOptions {
                    prompt: prompt.as_deref(),
                    force,
                    length,
                },
                &mut out,
                &mut err,
                &mut input,
            )
        }
        Command::Recommend => {
            let hw = commands::live_hardware();
            let meas = commands::live_measurements_dir();
            match commands::cmd_recommend(&catalog_dir, &meas, &hw, &mut out) {
                Ok(code) => code,
                Err(e) => fail(&e.0),
            }
        }
        Command::Serve { host, port } => {
            // One manager shared by the server (and, in-process, by
            // nothing else — but the same construction as `run`, so
            // a model loaded here is the same manager `run` uses).
            let manager = Arc::new(manager::ModelManager::new(
                &catalog_dir,
                &workers::workers_dir(),
                &commands::live_measurements_dir(),
            ));
            let state = Arc::new(server::ServerState {
                manager,
                catalog_dir: catalog_dir.clone(),
            });
            println!("ollada serving on http://{host}:{port}");
            let _ = out.flush();
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("error: could not start async runtime: {e}");
                    std::process::exit(1);
                }
            };
            match runtime.block_on(server::serve(state, &host, port)) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("error: {e}");
                    1
                }
            }
        }
        Command::Ps { port } => commands::cmd_ps(port, &mut out),
        Command::Rm { name } => match commands::cmd_rm(&catalog_dir, &name, &mut out) {
            Ok(code) => code,
            Err(e) => fail(&e.0),
        },
        Command::Workers { command } => {
            let runner = workers::RealRunner;
            let wdir = workers::workers_dir();
            match command {
                // No subcommand lists, like Python's argparse default.
                None | Some(WorkersCommand::List) => {
                    match commands::cmd_workers_list(&catalog_dir, &wdir, &runner, &mut out) {
                        Ok(code) => code,
                        Err(e) => fail(&e.0),
                    }
                }
                Some(WorkersCommand::Rebuild { name }) => {
                    match commands::cmd_workers_rebuild(
                        &catalog_dir,
                        &name,
                        &wdir,
                        &runner,
                        None,
                        &mut out,
                    ) {
                        Ok(code) => code,
                        Err(e) => fail(&e.0),
                    }
                }
            }
        }
    };
    let _ = out.flush();
    std::process::exit(code);
}

/// Mirrors Python's `main()` RecipeError handling: `error: ...` on
/// stderr, exit 2. (Exits directly: the error path never reaches the
/// normal flush/exit below, matching Python's `return 2`.)
fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}
