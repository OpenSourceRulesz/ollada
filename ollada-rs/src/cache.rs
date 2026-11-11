//! Hugging Face cache lookup for `rm` — the Rust equivalent of what the
//! Python CLI does with `huggingface_hub.scan_cache_dir`.
//!
//! Layout (same on Windows and Unix): `$HF_HOME/hub/` (default
//! `~/.cache/huggingface/hub/`) holds `models--<org>--<name>/` dirs.
//! A repo's on-disk size is the sum of the sizes of the files under its
//! `snapshots/` tree, following symlinks — which is exactly how
//! huggingface_hub computes `size_on_disk` (verified against the
//! sandbox cache: SDAR-4B-Chat sums to 11,623,344 bytes both ways).
//! If a repo ever has several snapshots, files shared between them are
//! counted once per snapshot, as huggingface_hub does.

use crate::measurements::home_dir;
use std::path::{Path, PathBuf};

/// The HF cache root: `$HF_HOME`, else `~/.cache/huggingface`.
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HF_HOME") {
        if !home.is_empty() {
            return PathBuf::from(home);
        }
    }
    home_dir().join(".cache").join("huggingface")
}

/// Sum of file sizes under `dir`, following symlinks (metadata follows
/// symlinks; a symlink to a blob reports the blob's size).
fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match std::fs::metadata(&path) {
                Ok(md) if md.is_dir() => stack.push(path),
                Ok(md) => total = total.saturating_add(md.len()),
                Err(_) => continue,
            }
        }
    }
    total
}

/// Mirrors huggingface_hub's `_format_size`: divide by 1000 while >= 1000,
/// format `{:.1f}` with width 3, then the unit letter.
pub fn format_size(num: u64) -> String {
    let mut num_f = num as f64;
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"] {
        if num_f.abs() < 1000.0 {
            return format!("{num_f:3.1}{unit}");
        }
        num_f /= 1000.0;
    }
    format!("{num_f:.1}Y")
}

/// Find a cached copy of `repo_id` (`org/name`) in the HF cache.
/// Returns its on-disk size in bytes when present.
pub fn find_cached_repo(repo_id: &str, cache_dir: &Path) -> Option<u64> {
    let dirname = format!("models--{}", repo_id.replace('/', "--"));
    let repo_dir = cache_dir.join("hub").join(dirname);
    if !repo_dir.is_dir() {
        return None;
    }
    let snapshots = repo_dir.join("snapshots");
    if !snapshots.is_dir() {
        return None;
    }
    Some(dir_size_bytes(&snapshots))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_matches_huggingface_hub() {
        // Values from huggingface_hub's _format_size, checked live.
        assert_eq!(format_size(0), "0.0");
        assert_eq!(format_size(512), "512.0");
        assert_eq!(format_size(1023), "1.0K");
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1536), "1.5K");
        assert_eq!(format_size(11_623_344), "11.6M");
        assert_eq!(format_size(4_075_086_435), "4.1G");
        assert_eq!(format_size(1_099_511_627_776), "1.1T");
    }

    #[test]
    fn find_cached_repo_matches_python_scan() {
        // The sandbox HF cache holds SDAR-4B-Chat; Python's
        // scan_cache_dir reports 11,623,344 bytes for it.
        let cache = hf_cache_dir();
        let size = find_cached_repo("JetLM/SDAR-4B-Chat", &cache);
        assert_eq!(size, Some(11_623_344));
        assert_eq!(find_cached_repo("org/nope", &cache), None);
    }
}
