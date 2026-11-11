//! Fit check — will this model actually fit on this machine?
//! A port of `ollada/fit.py`.
//!
//! The verdict is computed from, in order of trust:
//!   1. a LOCAL measurement — this machine, this GPU;
//!   2. the recipe's verified peak — ollada's own real-load number;
//!   3. the recipe's bf16 weights size — labelled an estimate, never a promise.
//!
//! HEADROOM_GB is reserved on top of the peak for the display driver, the
//! desktop, and whatever else shares the card.
//!
//! Detection difference from Python, documented: Python tries torch
//! first, then nvidia-smi. This crate has no torch, so it goes straight
//! to nvidia-smi, which ships with the NVIDIA driver on Windows and
//! Linux and is present wherever a GPU is. On a machine with torch but
//! no nvidia-smi in PATH, Python would detect a GPU and this crate
//! would not.

use crate::measurements::Record;
use crate::recipes::Recipe;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

pub const HEADROOM_GB: f64 = 1.0;

pub const FITS: &str = "fits";
pub const TIGHT: &str = "tight";
pub const WONT_FIT: &str = "won't fit";
pub const UNKNOWN: &str = "unknown";

#[derive(Debug, Clone, Default)]
pub struct Hardware {
    pub gpu_name: Option<String>,
    pub vram_total_gb: Option<f64>,
    pub vram_free_gb: Option<f64>,
    pub ram_total_gb: Option<f64>,
}

// `ram_floor_gb`, `ram_short`, and `blocking_reason` are ported API
// surface from ollada/fit.py: the Phase 0 CLI prints them only via
// `summary()`/`notes`, but the unit tests pin them, and Phase 1's
// load-refusal path will call `blocking_reason()` directly.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FitReport {
    pub model: String,
    pub verdict: String,
    pub peak_gb: Option<f64>,
    pub peak_source: String,
    pub required_gb: Option<f64>,
    pub hardware: Hardware,
    pub ram_floor_gb: Option<f64>,
    pub ram_short: bool,
    pub notes: Vec<String>,
}

#[allow(dead_code)]
impl FitReport {
    /// Why a load must be refused without --force, or None to proceed.
    pub fn blocking_reason(&self) -> Option<String> {
        let hw = &self.hardware;
        if self.verdict == WONT_FIT {
            if let (Some(total), Some(peak)) = (hw.vram_total_gb, self.peak_gb) {
                return Some(format!(
                    "{} measured a {:.2} GB peak ({}); this GPU ({}) has {:.1} GB total. \
                     Refusing to load — it would OOM mid-load or mid-generation. \
                     Override with --force if you know better.",
                    self.model,
                    peak,
                    self.peak_source,
                    hw.gpu_name.clone().unwrap_or_else(|| "unknown".to_string()),
                    total,
                ));
            }
        }
        if self.ram_short {
            if let (Some(floor), Some(ram)) = (self.ram_floor_gb, hw.ram_total_gb) {
                return Some(format!(
                    "{} declares a {:.0} GB host-RAM floor (recipe verified notes); \
                     this machine has {:.1} GB. Its load stages weights through host RAM \
                     and has failed on less. Override with --force if you know better.",
                    self.model, floor, ram,
                ));
            }
        }
        None
    }

    pub fn summary(&self) -> String {
        let peak = match self.peak_gb {
            None => {
                return format!(
                    "{}: fit unknown — no peak measurement on record",
                    self.model
                );
            }
            Some(p) => p,
        };
        let hw = &self.hardware;
        let gpu = hw
            .gpu_name
            .clone()
            .unwrap_or_else(|| "this GPU".to_string());
        let vram = match hw.vram_total_gb {
            Some(total) => format!("{total:.1} GB"),
            None => "unknown VRAM".to_string(),
        };
        format!(
            "{}: {} — peak {:.2} GB ({}), needs ~{:.1} GB with headroom; {} has {}",
            self.model,
            self.verdict,
            peak,
            self.peak_source,
            self.required_gb.unwrap_or(0.0),
            gpu,
            vram,
        )
    }
}

/// (peak_gb, source label) by trust order: local, verified, estimate.
pub fn peak_for(recipe: &Recipe, record: Option<&Record>) -> (Option<f64>, String) {
    if let Some(record) = record {
        if let Some(peak) = record.peak_vram_gb {
            let where_hw = record
                .hardware
                .clone()
                .unwrap_or_else(|| "this machine".to_string());
            return (Some(peak), format!("measured on this machine ({where_hw})"));
        }
    }
    if let Some(vram) = recipe.verified.get("vram_gb").and_then(|v| v.as_f64()) {
        let where_hw = recipe
            .verified
            .get("hardware")
            .and_then(|v| v.as_str())
            .unwrap_or("verified hardware");
        return (Some(vram), format!("ollada-verified on {where_hw}"));
    }
    if recipe.size_gb_bf16 > 0.0 {
        return (
            Some(recipe.size_gb_bf16),
            "weights-size estimate (unverified)".to_string(),
        );
    }
    (None, "no data".to_string())
}

/// Mirrors `ollada.fit.assess`.
pub fn assess(recipe: &Recipe, hardware: &Hardware, record: Option<&Record>) -> FitReport {
    let (peak, source) = peak_for(recipe, record);
    let floor = recipe
        .verified
        .get("min_host_ram_gb")
        .and_then(|v| v.as_f64());
    let ram_floor = floor;
    let ram_short = ram_floor.is_some()
        && hardware.ram_total_gb.is_some()
        && hardware.ram_total_gb.unwrap() < ram_floor.unwrap();
    // Python: required_gb = round(peak + HEADROOM_GB, 2).
    let required_gb = peak.map(|p| ((p + HEADROOM_GB) * 100.0).round() / 100.0);
    let mut report = FitReport {
        model: recipe.name.clone(),
        verdict: UNKNOWN.to_string(),
        peak_gb: peak,
        peak_source: source,
        required_gb,
        hardware: hardware.clone(),
        ram_floor_gb: ram_floor,
        ram_short,
        notes: Vec::new(),
    };
    if ram_short {
        report.notes.push(format!(
            "host RAM {:.1} GB is below this recipe's {:.0} GB floor",
            hardware.ram_total_gb.unwrap(),
            ram_floor.unwrap()
        ));
    }
    if peak.is_none() || hardware.vram_total_gb.is_none() {
        if hardware.vram_total_gb.is_none() {
            if hardware.gpu_name.is_some() {
                report.notes.push(
                    "GPU detected, but memory total unavailable — VRAM verdict unavailable"
                        .to_string(),
                );
            } else {
                report
                    .notes
                    .push("no GPU detected — VRAM verdict unavailable".to_string());
            }
        }
        return report;
    }
    let peak = peak.unwrap();
    let total = hardware.vram_total_gb.unwrap();
    let required = required_gb.unwrap();
    if peak > total {
        report.verdict = WONT_FIT.to_string();
    } else if required > total {
        report.verdict = TIGHT.to_string();
        report
            .notes
            .push("fits only with the card otherwise idle".to_string());
    } else if let Some(free) = hardware.vram_free_gb {
        if free < required {
            report.verdict = TIGHT.to_string();
            report.notes.push(format!(
                "fits the card, but only {free:.1} GB is free right now"
            ));
        } else {
            report.verdict = FITS.to_string();
        }
    } else {
        report.verdict = FITS.to_string();
    }
    report
}

// -- hardware detection ------------------------------------------------------

/// Total host RAM in GB, via sysinfo (cross-platform: Linux, macOS, Windows).
fn ram_total_gb() -> Option<f64> {
    let sys = match sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    ) {
        Ok(sys) => sys,
        Err(_) => return None,
    };
    let bytes = sys.total_memory();
    if bytes == 0 {
        None
    } else {
        Some(bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Outcome of parsing one nvidia-smi CSV line.
#[derive(Debug, PartialEq)]
enum SmiLine {
    /// `name, total_mib, free_mib` all parsed.
    Parsed {
        name: String,
        total_gb: f64,
        free_gb: f64,
    },
    /// Memory fields are `[N/A]` (unified-memory GPUs such as GB10):
    /// the GPU is detected, VRAM stays unknown.
    UnifiedMemory { name: String },
    /// Not a data line (warning text, blank, wrong field count).
    Skip,
}

/// Parse one line of `nvidia-smi --format=csv,noheader,nounits` output.
/// Mirrors the loop body of `ollada.fit._detect_via_nvidia_smi`.
fn parse_smi_line(line: &str) -> SmiLine {
    let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
    if parts.len() != 3 {
        return SmiLine::Skip;
    }
    match (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {
        (Ok(total_mib), Ok(free_mib)) => SmiLine::Parsed {
            name: parts[0].to_string(),
            total_gb: total_mib / 1024.0,
            free_gb: free_mib / 1024.0,
        },
        _ => {
            if !parts[0].is_empty() && parts[0].starts_with("NVIDIA ") {
                SmiLine::UnifiedMemory {
                    name: parts[0].to_string(),
                }
            } else {
                SmiLine::Skip
            }
        }
    }
}

/// Run nvidia-smi with a 15 s timeout, mirroring
/// `ollada.fit._detect_via_nvidia_smi`. Returns false when nvidia-smi is
/// missing, fails, or yields no parseable line. std has no process
/// timeout, so the wait runs on a thread and the process is killed on
/// timeout — the same code path on Windows and Unix.
fn detect_via_nvidia_smi(hw: &mut Hardware) -> bool {
    let mut child = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false, // no nvidia-smi on PATH — no GPU to detect
    };
    let pid = child.id();
    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => return false,
    };
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        let _ = child.wait();
        let _ = tx.send(buf);
    });
    let output = match rx.recv_timeout(Duration::from_secs(15)) {
        Ok(buf) => buf,
        Err(_) => {
            // Timed out: kill the child by pid (sysinfo is cross-platform)
            // so the reader thread can finish, and give up on GPU detection.
            let mut sys = match sysinfo::System::new() {
                Ok(sys) => sys,
                Err(_) => return false,
            };
            sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
            if let Some(process) = sys.process(sysinfo::Pid::from_u32(pid)) {
                let _ = process.kill();
            }
            return false;
        }
    };
    let text = String::from_utf8_lossy(&output);
    // Take the first line that actually parses — a warning line ahead
    // of the data must not poison the whole detection.
    for line in text.lines() {
        match parse_smi_line(line) {
            SmiLine::Parsed {
                name,
                total_gb,
                free_gb,
            } => {
                hw.gpu_name = Some(name);
                hw.vram_total_gb = Some(total_gb);
                hw.vram_free_gb = Some(free_gb);
                return true;
            }
            SmiLine::UnifiedMemory { name } => {
                hw.gpu_name = Some(name);
                return true;
            }
            SmiLine::Skip => continue,
        }
    }
    false
}

/// Mirrors `ollada.fit.detect_hardware` (without the torch path).
pub fn detect_hardware() -> Hardware {
    let mut hw = Hardware {
        ram_total_gb: ram_total_gb(),
        ..Default::default()
    };
    detect_via_nvidia_smi(&mut hw);
    hw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::validate_recipe;
    use serde_yaml::Value as YamlValue;

    fn make_recipe(verified_yaml: &str) -> Recipe {
        let yaml = format!(
            "name: 'fake:1b'\nfamily: fake\nsource: org/fake\nlicense: MIT\ntask: chat\n\
             description: ''\nsize_gb_bf16: 2.0\nquant_default: bf16\ndtype: float16\n\
             generate: {{steps: 1}}\nverified: {verified_yaml}\n"
        );
        let data: YamlValue = serde_yaml::from_str(&yaml).unwrap();
        validate_recipe(&data, None).unwrap()
    }

    fn hw() -> Hardware {
        Hardware {
            gpu_name: Some("Test GPU".to_string()),
            vram_total_gb: Some(16.0),
            vram_free_gb: Some(15.0),
            ram_total_gb: Some(32.0),
        }
    }

    // -- nvidia-smi parsing ---------------------------------------------------

    #[test]
    fn smi_line_parses_a_normal_gpu() {
        assert_eq!(
            parse_smi_line("NVIDIA GeForce RTX 5060 Ti, 16346, 15000"),
            SmiLine::Parsed {
                name: "NVIDIA GeForce RTX 5060 Ti".to_string(),
                total_gb: 16346.0 / 1024.0,
                free_gb: 15000.0 / 1024.0,
            }
        );
    }

    #[test]
    fn smi_line_skips_warnings_and_garbage() {
        assert_eq!(parse_smi_line(""), SmiLine::Skip);
        assert_eq!(parse_smi_line("WARNING: something odd"), SmiLine::Skip);
        assert_eq!(parse_smi_line("only, two"), SmiLine::Skip);
    }

    #[test]
    fn smi_line_detects_unified_memory_gpu() {
        // GB10 (PR #1): memory fields report [N/A]; the GPU is still
        // detected so workers install CUDA torch, VRAM stays unknown.
        assert_eq!(
            parse_smi_line("NVIDIA GB10, [N/A], [N/A]"),
            SmiLine::UnifiedMemory {
                name: "NVIDIA GB10".to_string()
            }
        );
    }

    #[test]
    fn smi_line_does_not_claim_a_non_nvidia_na_line() {
        assert_eq!(
            parse_smi_line("Some Other Card, [N/A], [N/A]"),
            SmiLine::Skip
        );
    }

    // -- peak trust order ------------------------------------------------------

    #[test]
    fn local_measurement_beats_recipe_verified() {
        let recipe = make_recipe("{vram_gb: 9.8, hardware: 'Tesla T4', date: '2026-10-21'}");
        let record = Record {
            model: Some("fake:1b".to_string()),
            peak_vram_gb: Some(10.4),
            observations: Some(3),
            hardware: Some("RTX Test".to_string()),
            updated: Some("2026-11-01".to_string()),
        };
        let (peak, source) = peak_for(&recipe, Some(&record));
        assert_eq!(peak, Some(10.4));
        assert!(source.contains("this machine"));
    }

    #[test]
    fn recipe_verified_is_second() {
        let recipe = make_recipe("{vram_gb: 8.87, hardware: 'Tesla T4', date: '2026-10-21'}");
        let (peak, source) = peak_for(&recipe, None);
        assert_eq!(peak, Some(8.87));
        assert!(source.contains("Tesla T4"));
    }

    #[test]
    fn unverified_falls_back_to_labelled_estimate() {
        let recipe = make_recipe("{}");
        let (peak, source) = peak_for(&recipe, None);
        assert_eq!(peak, Some(2.0));
        assert!(source.contains("estimate") && source.contains("unverified"));
    }

    // -- verdict matrix (mirrors tests/test_fit.py) -----------------------------

    #[test]
    fn fits_with_headroom() {
        let report = assess(&make_recipe("{vram_gb: 9.8}"), &hw(), None);
        assert_eq!(report.verdict, FITS);
        assert!(report.blocking_reason().is_none());
        assert_eq!(report.required_gb, Some(10.8));
    }

    #[test]
    fn tight_when_only_the_headroom_overflows() {
        // The DiffCoder-on-T4 case: 15.15 GB peak on a 15.6 GB card.
        let mut h = hw();
        h.vram_total_gb = Some(15.6);
        let report = assess(&make_recipe("{vram_gb: 15.15}"), &h, None);
        assert_eq!(report.verdict, TIGHT);
        assert!(report.blocking_reason().is_none()); // tight warns; it does not refuse
    }

    #[test]
    fn wont_fit_and_blocks() {
        let mut h = hw();
        h.vram_total_gb = Some(12.0);
        let report = assess(&make_recipe("{vram_gb: 15.15}"), &h, None);
        assert_eq!(report.verdict, WONT_FIT);
        let reason = report.blocking_reason().unwrap();
        assert!(reason.contains("--force"));
    }

    #[test]
    fn tight_when_free_vram_is_low() {
        let mut h = hw();
        h.vram_total_gb = Some(24.0);
        h.vram_free_gb = Some(10.0);
        let report = assess(&make_recipe("{vram_gb: 9.8}"), &h, None);
        assert_eq!(report.verdict, TIGHT);
    }

    #[test]
    fn unknown_without_gpu_never_blocks() {
        let no_gpu = Hardware {
            ram_total_gb: Some(32.0),
            ..Default::default()
        };
        let report = assess(&make_recipe("{vram_gb: 99.0}"), &no_gpu, None);
        assert_eq!(report.verdict, UNKNOWN);
        assert!(report.blocking_reason().is_none());
        assert!(report.notes[0].contains("no GPU detected"));
    }

    #[test]
    fn gb10_reports_gpu_detected_but_no_vram_verdict() {
        let gb10 = Hardware {
            gpu_name: Some("NVIDIA GB10".to_string()),
            ram_total_gb: Some(121.7),
            ..Default::default()
        };
        let report = assess(&make_recipe("{}"), &gb10, None);
        assert_eq!(report.verdict, UNKNOWN);
        assert!(report.notes[0].contains("GPU detected"));
    }

    #[test]
    fn ram_floor_blocks_only_when_declared() {
        let recipe = make_recipe("{min_host_ram_gb: 16, hardware: 'T4', date: '2026-10-21'}");
        let mut h = hw();
        h.ram_total_gb = Some(8.0);
        let report = assess(&recipe, &h, None);
        assert!(report.ram_short);
        assert!(report.blocking_reason().is_some());
        let no_floor = make_recipe("{vram_gb: 9.8}");
        assert!(assess(&no_floor, &h, None).blocking_reason().is_none());
    }

    #[test]
    fn summary_matches_python_format() {
        let report = assess(
            &make_recipe("{vram_gb: 9.8, hardware: 'Tesla T4'}"),
            &hw(),
            None,
        );
        assert_eq!(
            report.summary(),
            "fake:1b: fits — peak 9.80 GB (ollada-verified on Tesla T4), needs ~10.8 GB with headroom; Test GPU has 16.0 GB"
        );
    }
}
