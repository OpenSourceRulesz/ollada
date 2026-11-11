//! Recipe loading and validation (schema v1) — a port of `ollada/recipes.py`.
//!
//! A Recipe is everything a user should never have to guess: source repo,
//! quantization, dtype, dependency pins, and generation defaults.

use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// Mirrors `ollada.recipes.RecipeError`.
#[derive(Debug)]
pub struct RecipeError(pub String);

impl fmt::Display for RecipeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RecipeError {}

const REQUIRED_FIELDS: [&str; 9] = [
    "name",
    "family",
    "source",
    "license",
    "task",
    "size_gb_bf16",
    "quant_default",
    "dtype",
    "generate",
];

// `path` is ported API surface (Python's Recipe carries it); the
// Phase 0 CLI does not print it.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Recipe {
    pub name: String,
    pub family: String,
    pub source: String,
    pub license: String,
    pub task: String,
    pub description: String,
    pub size_gb_bf16: f64,
    pub quant_default: String,
    pub dtype: String,
    /// Raw YAML mapping, kept so `show` can print it with Python repr
    /// in the original key order.
    pub generate: YamlValue,
    pub env: YamlValue,
    pub verified: YamlValue,
    pub path: Option<PathBuf>,
}

impl Recipe {
    /// A model counts as verified only with a real hardware run on record.
    pub fn is_verified(&self) -> bool {
        let hardware = self
            .verified
            .get("hardware")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let date = self
            .verified
            .get("date")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        !hardware.is_empty() && !date.is_empty()
    }

    /// The `verified.hardware` string, or `"no"` for the `list` column.
    pub fn verified_hardware(&self) -> String {
        match self.verified.get("hardware") {
            Some(YamlValue::String(s)) if !s.is_empty() => s.clone(),
            _ => "no".to_string(),
        }
    }

    /// Mirrors `ollada.recipes.Recipe.to_public_dict`: the
    /// Ollama-shaped `/api/tags` entry, in Python's key order.
    /// `size_gb_bf16` keeps its float shape (`4.1`, `16.0`); the
    /// `verified` flag is a bool, not the raw mapping.
    pub fn to_public_dict(&self) -> JsonValue {
        let mut map = serde_json::Map::new();
        map.insert("name".to_string(), JsonValue::String(self.name.clone()));
        map.insert("model".to_string(), JsonValue::String(self.name.clone()));
        map.insert("family".to_string(), JsonValue::String(self.family.clone()));
        map.insert("source".to_string(), JsonValue::String(self.source.clone()));
        map.insert(
            "license".to_string(),
            JsonValue::String(self.license.clone()),
        );
        map.insert("task".to_string(), JsonValue::String(self.task.clone()));
        map.insert(
            "description".to_string(),
            JsonValue::String(self.description.clone()),
        );
        map.insert(
            "size_gb_bf16".to_string(),
            serde_json::Number::from_f64(self.size_gb_bf16)
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        );
        map.insert(
            "quant_default".to_string(),
            JsonValue::String(self.quant_default.clone()),
        );
        map.insert("verified".to_string(), JsonValue::Bool(self.is_verified()));
        JsonValue::Object(map)
    }
}

fn valid_name(name: &str) -> bool {
    // ^[a-z0-9][a-z0-9\-]*:[a-z0-9][a-z0-9\.]*$
    let mut parts = name.splitn(2, ':');
    let head = parts.next().unwrap_or("");
    let tail = match parts.next() {
        Some(t) => t,
        None => return false,
    };
    if head.is_empty() || tail.is_empty() {
        return false;
    }
    let mut head_chars = head.chars();
    match head_chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    if !head_chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        return false;
    }
    if !tail
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.')
    {
        return false;
    }
    true
}

/// Mirrors `ollada.recipes.validate_recipe`.
pub fn validate_recipe(data: &YamlValue, path: Option<&Path>) -> Result<Recipe, RecipeError> {
    let where_suffix = match path {
        Some(p) => format!(" in {}", p.display()),
        None => String::new(),
    };
    let map = match data {
        YamlValue::Mapping(m) => m,
        _ => {
            return Err(RecipeError(format!(
                "recipe is not a mapping{where_suffix}"
            )))
        }
    };
    for key in REQUIRED_FIELDS {
        if !map.contains_key(YamlValue::String(key.to_string())) {
            return Err(RecipeError(format!(
                "recipe missing required field '{key}'{where_suffix}"
            )));
        }
    }
    let get_str = |key: &str| -> String {
        match map.get(YamlValue::String(key.to_string())) {
            Some(YamlValue::String(s)) => s.clone(),
            Some(other) => yaml_scalar_string(other),
            None => String::new(),
        }
    };
    let name = get_str("name");
    if !valid_name(&name) {
        return Err(RecipeError(format!(
            "invalid model name '{name}' (want e.g. 'dream:7b'){where_suffix}"
        )));
    }
    let generate = map
        .get(YamlValue::String("generate".to_string()))
        .cloned()
        .unwrap_or(YamlValue::Null);
    match &generate {
        YamlValue::Mapping(m) if !m.is_empty() => {}
        _ => {
            return Err(RecipeError(format!(
                "'generate' must be a non-empty mapping{where_suffix}"
            )))
        }
    }
    let size_value = map
        .get(YamlValue::String("size_gb_bf16".to_string()))
        .cloned()
        .unwrap_or(YamlValue::Null);
    let size = match &size_value {
        YamlValue::Number(n) => n.as_f64(),
        _ => None,
    };
    let size = match size {
        Some(s) => s,
        None => {
            return Err(RecipeError(format!(
                "'size_gb_bf16' must be a number{where_suffix}"
            )))
        }
    };
    if size <= 0.0 {
        return Err(RecipeError(format!(
            "'size_gb_bf16' must be positive{where_suffix}"
        )));
    }
    let empty_map = YamlValue::Mapping(Default::default());
    let env = map
        .get(YamlValue::String("env".to_string()))
        .cloned()
        .unwrap_or_else(|| empty_map.clone());
    let env = if env.is_null() {
        empty_map.clone()
    } else {
        env
    };
    let verified = map
        .get(YamlValue::String("verified".to_string()))
        .cloned()
        .unwrap_or_else(|| empty_map.clone());
    let verified = if verified.is_null() {
        empty_map
    } else {
        verified
    };
    Ok(Recipe {
        name,
        family: get_str("family"),
        source: get_str("source"),
        license: get_str("license"),
        task: get_str("task"),
        description: get_str("description"),
        size_gb_bf16: size,
        quant_default: get_str("quant_default"),
        dtype: get_str("dtype"),
        generate,
        env,
        verified,
        path: path.map(|p| p.to_path_buf()),
    })
}

/// Best-effort scalar rendering for non-string YAML values in string fields.
fn yaml_scalar_string(value: &YamlValue) -> String {
    match value {
        YamlValue::Null => String::new(),
        YamlValue::Bool(b) => b.to_string(),
        YamlValue::Number(n) => n.to_string(),
        YamlValue::String(s) => s.clone(),
        _ => String::new(),
    }
}

/// Mirrors `ollada.recipes._version_tuple`: numeric prefix of a dotted
/// version (`4.57.6` -> (4, 57, 6)). Suffixes (`rc1`, `dev`) are ignored.
// `spec_satisfied` (and its helpers) are ported API surface used by
// the unit tests; the Phase 0 CLI does not check transformer pins
// (that is the manager's job, Phase 3).
#[allow(dead_code)]
fn version_tuple(version: &str) -> Vec<u64> {
    version
        .trim()
        .split('.')
        .map(|segment| {
            let digits: String = segment.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u64>().unwrap_or(0)
        })
        .collect()
}

#[allow(dead_code)]
fn pad(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = a.len().max(b.len());
    let mut a2 = a.to_vec();
    let mut b2 = b.to_vec();
    a2.resize(n, 0);
    b2.resize(n, 0);
    (a2, b2)
}

/// Mirrors `ollada.recipes.spec_satisfied`: true if `version` satisfies a
/// comma-separated specifier (`==`, `!=`, `>=`, `<=`, `>`, `<`, or a bare
/// version read as `==`). An unparseable clause fails closed.
#[allow(dead_code)]
pub fn spec_satisfied(spec: &str, version: &str) -> bool {
    let want = version_tuple(version);
    for clause in spec.split(',') {
        let clause = clause.trim();
        let (op, pinned) = if let Some(rest) = clause.strip_prefix("==") {
            ("==", rest)
        } else if let Some(rest) = clause.strip_prefix("!=") {
            ("!=", rest)
        } else if let Some(rest) = clause.strip_prefix(">=") {
            (">=", rest)
        } else if let Some(rest) = clause.strip_prefix("<=") {
            ("<=", rest)
        } else if let Some(rest) = clause.strip_prefix('>') {
            (">", rest)
        } else if let Some(rest) = clause.strip_prefix('<') {
            ("<", rest)
        } else {
            ("==", clause)
        };
        let pinned = pinned.trim();
        // ^[0-9][0-9A-Za-z.]*$ — the spec regex, minus its whitespace
        // handling (already trimmed above).
        if pinned.is_empty()
            || !pinned
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
            || !pinned
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.')
        {
            return false;
        }
        let (have, pin) = pad(&want, &version_tuple(pinned));
        let ok = match op {
            "==" => have == pin,
            "!=" => have != pin,
            ">=" => have >= pin,
            "<=" => have <= pin,
            ">" => have > pin,
            "<" => have < pin,
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Mirrors `ollada.recipes.load_catalog`: every recipe on the shelf,
/// sorted by name. Fails loudly on duplicates.
pub fn load_catalog(directory: &Path) -> Result<Vec<Recipe>, RecipeError> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(directory).map_err(|e| {
        RecipeError(format!(
            "cannot read catalog directory {}: {e}",
            directory.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| RecipeError(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            paths.push(path);
        }
    }
    paths.sort();
    let mut recipes = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for path in &paths {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RecipeError(format!("cannot read {}: {e}", path.display())))?;
        let data: YamlValue = serde_yaml::from_str(&text)
            .map_err(|e| RecipeError(format!("invalid YAML in {}: {e}", path.display())))?;
        let recipe = validate_recipe(&data, Some(path))?;
        if !seen.insert(recipe.name.clone()) {
            return Err(RecipeError(format!(
                "duplicate model name '{}' in catalog",
                recipe.name
            )));
        }
        recipes.push(recipe);
    }
    recipes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(recipes)
}

/// Mirrors `ollada.recipes.get_recipe`.
pub fn get_recipe(name: &str, directory: &Path) -> Result<Recipe, RecipeError> {
    let recipes = load_catalog(directory)?;
    for recipe in &recipes {
        if recipe.name == name {
            return Ok(recipe.clone());
        }
    }
    let available = recipes
        .iter()
        .map(|r| r.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let available = if available.is_empty() {
        "(empty shelf)".to_string()
    } else {
        available
    };
    Err(RecipeError(format!(
        "model '{name}' is not on the shelf. Available: {available}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(s: &str) -> YamlValue {
        serde_yaml::from_str(s).unwrap()
    }

    #[test]
    fn spec_satisfied_matches_python_cases() {
        // From tests/test_recipes.py: Dream's pin admits 4.53.3 and
        // 4.57.6, excludes 5.x; SDAR's ">=4.57,<5" admits 4.57.6.
        assert!(spec_satisfied(">=4.53.3,<5", "4.53.3"));
        assert!(spec_satisfied(">=4.53.3,<5", "4.57.6"));
        assert!(!spec_satisfied(">=4.53.3,<5", "5.3.0"));
        assert!(spec_satisfied(">=4.57,<5", "4.57.6"));
        assert!(!spec_satisfied(">=4.57,<5", "5.3.0"));
        assert!(spec_satisfied("==5.3.0", "5.3.0"));
        assert!(!spec_satisfied("==5.3.0", "5.20.2"));
        // Bare version reads as ==; suffixes are ignored (but 4.57.6rc1
        // is still version 4.57.6, not 4.57 — checked against Python).
        assert!(spec_satisfied("4.57.6", "4.57.6"));
        assert!(spec_satisfied("==4.57.6", "4.57.6rc1"));
        assert!(!spec_satisfied("==4.57", "4.57.6rc1"));
        // Unparseable clause fails closed.
        assert!(!spec_satisfied("banana", "4.57.6"));
        assert!(!spec_satisfied(">=x", "4.57.6"));
    }

    #[test]
    fn validate_rejects_missing_fields_and_bad_names() {
        let err = validate_recipe(&data("{name: 'dream:7b'}"), None).unwrap_err();
        assert!(err.0.contains("missing required field"));
        let bad = data(
            "name: 'No Colon'\nfamily: dream\nsource: x/y\nlicense: MIT\ntask: chat\nsize_gb_bf16: 1.0\nquant_default: int8\ndtype: float16\ngenerate: {steps: 1}\n",
        );
        assert!(validate_recipe(&bad, None).is_err());
    }

    #[test]
    fn validate_rejects_non_positive_size() {
        let bad = data(
            "name: 'dream:7b'\nfamily: dream\nsource: x/y\nlicense: MIT\ntask: chat\nsize_gb_bf16: 0\nquant_default: int8\ndtype: float16\ngenerate: {steps: 1}\n",
        );
        assert!(validate_recipe(&bad, None).is_err());
    }

    #[test]
    fn validate_accepts_a_good_recipe() {
        let good = data(
            "name: 'dream:7b'\nfamily: dream\nsource: x/y\nlicense: MIT\ntask: chat\nsize_gb_bf16: 15.2\nquant_default: int8\ndtype: float16\ngenerate: {steps: 128}\nverified: {hardware: 'T4', date: '2026-10-21'}\n",
        );
        let r = validate_recipe(&good, None).unwrap();
        assert_eq!(r.name, "dream:7b");
        assert!(r.is_verified());
    }

    #[test]
    fn names_follow_the_recipe_regex() {
        assert!(valid_name("dream:7b"));
        assert!(valid_name("diffcoder:8b4bit"));
        assert!(valid_name("sdar:1.7b"));
        assert!(!valid_name("No Colon"));
        assert!(!valid_name("dream"));
        assert!(!valid_name("dream:8b-int4")); // hyphen after the colon is illegal
        assert!(!valid_name(":7b"));
        assert!(!valid_name("Dream:7b"));
    }
}
