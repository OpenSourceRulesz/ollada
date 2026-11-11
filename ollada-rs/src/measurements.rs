//! Local measurement records — a port of `ollada/measurements.py`.
//!
//! Records live in `~/.ollada/measurements/` (or `$OLLADA_MEASUREMENTS_DIR`)
//! and never leave the machine. A record:
//! `{"model", "peak_vram_gb", "observations", "hardware", "updated"}`.
//! Phase 3 adds the write side (`record_peak`): the manager records the
//! worker-reported peak after each load and generation. There is no
//! `capture_and_record` here — that reads the peak from the manager's
//! own in-process torch, and the Rust manager has no in-process path
//! (see `manager.rs`): a worker-reported peak of `None` (CPU, no CUDA)
//! simply records nothing, which is what Python's capture returns on
//! a CPU box too.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Peaks below this are ignored: a model that never touched the GPU
/// (CPU fallback, failed load) must not write a flattering fake number.
/// Mirrors `ollada.measurements._MIN_PLAUSIBLE_GB`.
pub const MIN_PLAUSIBLE_GB: f64 = 0.05;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Record {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub peak_vram_gb: Option<f64>,
    pub observations: Option<u64>,
    pub hardware: Option<String>,
    #[allow(dead_code)]
    pub updated: Option<String>,
}

/// Mirrors `ollada.measurements.measurements_dir`.
pub fn measurements_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLADA_MEASUREMENTS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    home_dir().join(".ollada").join("measurements")
}

/// The user's home directory: `$HOME` on Unix, `%USERPROFILE%`
/// (falling back to `%HOMEDRIVE%%HOMEPATH%`) on Windows.
pub fn home_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home);
        }
    }
    if let Ok(profile) = std::env::var("USERPROFILE") {
        if !profile.is_empty() {
            return PathBuf::from(profile);
        }
    }
    if let (Ok(drive), Ok(path)) = (std::env::var("HOMEDRIVE"), std::env::var("HOMEPATH")) {
        return PathBuf::from(format!("{drive}{path}"));
    }
    PathBuf::from(".")
}

fn record_path(model: &str, directory: &std::path::Path) -> PathBuf {
    let safe: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    directory.join(format!("{safe}.json"))
}

/// Mirrors `ollada.measurements.load_record`: the local measurement
/// record for a model, or None if never measured here (or unreadable).
pub fn load_record(model: &str, directory: &std::path::Path) -> Option<Record> {
    let text = std::fs::read_to_string(record_path(model, directory)).ok()?;
    let record: Record = serde_json::from_str(&text).ok()?;
    record.peak_vram_gb?; // a record without a peak is not a record
    Some(record)
}

/// Today's date as `YYYY-MM-DD`, like Python's
/// `time.strftime("%Y-%m-%d")`. Std has no date formatting, so this
/// converts the Unix time with a civil-date algorithm (Howard
/// Hinnant's, days since the epoch). Deliberate difference: this is
/// the UTC day, Python's is the local day — the two differ only for
/// runs near midnight UTC, and `updated` is advisory only.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    // Civil from days (Hinnant): days since 1970-01-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}")
}

/// Mirrors `ollada.measurements.record_peak`: merge one observed peak
/// into the model's record (max wins). Returns the record after the
/// merge, or the existing record when the peak is implausible
/// (< 0.05 GB) or `None`. Measurements are advisory: a write failure
/// returns the existing record instead of breaking the run.
pub fn record_peak(
    model: &str,
    peak_vram_gb: Option<f64>,
    hardware: Option<&str>,
    directory: &std::path::Path,
) -> Option<Record> {
    let peak = match peak_vram_gb {
        Some(p) => p,
        None => return load_record(model, directory),
    };
    if peak < MIN_PLAUSIBLE_GB {
        return load_record(model, directory);
    }
    let existing = load_record(model, directory);
    let record = Record {
        model: Some(model.to_string()),
        peak_vram_gb: Some(match existing.as_ref().and_then(|r| r.peak_vram_gb) {
            Some(old) => peak.max(old),
            None => peak,
        }),
        observations: Some(match existing.as_ref().and_then(|r| r.observations) {
            Some(n) => n + 1,
            None => 1,
        }),
        hardware: match hardware {
            Some(h) => Some(h.to_string()),
            None => existing.as_ref().and_then(|r| r.hardware.clone()),
        },
        updated: Some(today()),
    };
    if write_record(model, directory, &record).is_err() {
        return existing;
    }
    Some(record)
}

/// Atomic write of one record: write the `.tmp` sibling, then rename
/// over the record — mirrors Python's `tmp.replace(path)`. The JSON
/// is dumped with keys sorted, like Python's `sort_keys=True`.
fn write_record(
    model: &str,
    directory: &std::path::Path,
    record: &Record,
) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(directory)?;
    let path = record_path(model, directory);
    let tmp = path.with_extension("tmp");
    let mut map = serde_json::Map::new();
    map.insert(
        "hardware".to_string(),
        record
            .hardware
            .clone()
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "model".to_string(),
        record
            .model
            .clone()
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "observations".to_string(),
        record
            .observations
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "peak_vram_gb".to_string(),
        record
            .peak_vram_gb
            .map(|p| {
                serde_json::Number::from_f64(p)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            })
            .unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "updated".to_string(),
        record
            .updated
            .clone()
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(map))
        .map_err(std::io::Error::other)?;
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_record_roundtrip_and_missing() {
        let dir = std::env::temp_dir().join("ollada-rs-test-measurements");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(load_record("dream:7b", &dir).is_none());
        std::fs::write(
            dir.join("dream_7b.json"),
            r#"{"model": "dream:7b", "peak_vram_gb": 9.8, "observations": 2, "hardware": "T4", "updated": "2026-10-21"}"#,
        )
        .unwrap();
        let record = load_record("dream:7b", &dir).unwrap();
        assert_eq!(record.peak_vram_gb, Some(9.8));
        assert_eq!(record.observations, Some(2));
        // A record without a peak is not a record (mirrors Python).
        std::fs::write(dir.join("sdar_1.7b.json"), r#"{"model": "sdar:1.7b"}"#).unwrap();
        assert!(load_record("sdar:1.7b", &dir).is_none());
        // Malformed JSON is not a record either.
        std::fs::write(dir.join("nemotron_3b.json"), "not json").unwrap();
        assert!(load_record("nemotron:3b", &dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_path_sanitises_the_model_name() {
        let dir = PathBuf::from("/tmp/x");
        assert_eq!(
            record_path("dream:7b", &dir),
            PathBuf::from("/tmp/x/dream_7b.json")
        );
    }

    fn peak_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ollada-rs-test-record-peak-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn record_peak_creates_max_wins_and_counts_observations() {
        let dir = peak_dir("max-wins");
        let first = record_peak("dream:7b", Some(9.8), Some("T4"), &dir).unwrap();
        assert_eq!(first.peak_vram_gb, Some(9.8));
        assert_eq!(first.observations, Some(1));
        assert_eq!(first.hardware.as_deref(), Some("T4"));
        // A lower peak does not move the record, but still counts.
        let second = record_peak("dream:7b", Some(9.1), None, &dir).unwrap();
        assert_eq!(second.peak_vram_gb, Some(9.8));
        assert_eq!(second.observations, Some(2));
        assert_eq!(second.hardware.as_deref(), Some("T4")); // kept
                                                            // A higher peak wins, and a new hardware string replaces it.
        let third = record_peak("dream:7b", Some(10.4), Some("A100"), &dir).unwrap();
        assert_eq!(third.peak_vram_gb, Some(10.4));
        assert_eq!(third.observations, Some(3));
        assert_eq!(third.hardware.as_deref(), Some("A100"));
        // The file on disk is sorted-key JSON, like Python's
        // json.dump(indent=2, sort_keys=True), and reloads.
        let text = std::fs::read_to_string(dir.join("dream_7b.json")).unwrap();
        assert!(text.starts_with("{\n  \"hardware\": \"A100\","));
        assert!(text.contains("\"peak_vram_gb\": 10.4"));
        assert_eq!(load_record("dream:7b", &dir).unwrap().observations, Some(3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_peak_ignores_implausible_and_missing_peaks() {
        let dir = peak_dir("implausible");
        // Below 0.05 GB: no record is created (CPU fallback must not
        // write a flattering fake number).
        assert!(record_peak("sdar:1.7b", Some(0.01), None, &dir).is_none());
        assert!(!dir.join("sdar_1.7b.json").exists());
        // None: also no record — and with an existing record, the
        // existing record is returned unchanged.
        assert!(record_peak("sdar:1.7b", None, None, &dir).is_none());
        record_peak("sdar:1.7b", Some(4.1), Some("CPU"), &dir).unwrap();
        let kept = record_peak("sdar:1.7b", None, None, &dir).unwrap();
        assert_eq!(kept.peak_vram_gb, Some(4.1));
        assert_eq!(kept.observations, Some(1));
        let kept2 = record_peak("sdar:1.7b", Some(0.01), None, &dir).unwrap();
        assert_eq!(kept2.observations, Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_peak_survives_a_malformed_existing_record() {
        let dir = peak_dir("malformed");
        std::fs::write(dir.join("dream_7b.json"), "not json").unwrap();
        let record = record_peak("dream:7b", Some(9.8), None, &dir).unwrap();
        assert_eq!(record.peak_vram_gb, Some(9.8));
        assert_eq!(record.observations, Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
