//! Python `repr()` formatting for values the CLI prints.
//!
//! The Python CLI prints dicts with `repr` (e.g. `show` prints the
//! `generate:` and `verified:` mappings, and `ps` prints the daemon's
//! JSON as a Python dict). To match its output byte-for-byte, this module
//! reproduces the relevant parts of Python's repr:
//!   - strings: single quotes unless the string contains a single quote
//!     (then double quotes, unless it also contains one — then single
//!     quotes with the inner ones escaped), with `\n`, `\t`, `\r`, `\\`
//!     and control characters escaped.
//!   - floats: Rust's `{:?}` matches Python's repr for the magnitudes
//!     ollada prints (GB sizes, thresholds); both give `16.0`, `6.59`,
//!     `0.9`. Very large/small magnitudes can differ (`1e22` vs
//!     `1e+22`); ollada never prints those.
//!   - bool: `True` / `False`; null: `None`.
//!   - mappings: `{'key': value, ...}` in insertion order.

use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

/// Python `str(float)` / `repr(float)` for the magnitudes ollada prints.
pub fn py_float(value: f64) -> String {
    // Rust's Debug for f64 prints the shortest round-tripping decimal
    // with a `.0` for whole values, exactly like Python's repr here.
    // NaN/inf never occur in recipes; fall back to str() behaviour.
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        format!("{value:?}")
    }
}

/// Python `repr()` of a string.
pub fn py_str(value: &str) -> String {
    let has_single = value.contains('\'');
    let has_double = value.contains('"');
    let (quote, escape_other): (char, bool) = if has_single && !has_double {
        ('"', false)
    } else {
        ('\'', true)
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\'' if quote == '\'' => out.push_str("\\'"),
            '"' if quote == '"' => out.push_str("\\\""),
            '"' if escape_other => out.push('"'),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_yaml_scalar(value: &YamlValue) -> String {
    match value {
        YamlValue::Null => "None".to_string(),
        YamlValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        YamlValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                py_float(f)
            } else {
                n.to_string()
            }
        }
        YamlValue::String(s) => py_str(s),
        YamlValue::Sequence(seq) => {
            let items: Vec<String> = seq.iter().map(py_yaml).collect();
            format!("[{}]", items.join(", "))
        }
        YamlValue::Mapping(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_yaml(k), py_yaml(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        YamlValue::Tagged(t) => py_yaml(&t.value),
    }
}

/// Python `repr()` of a YAML value (as parsed by serde_yaml).
pub fn py_yaml(value: &YamlValue) -> String {
    py_yaml_scalar(value)
}

fn py_json_scalar(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "None".to_string(),
        JsonValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                py_float(f)
            } else {
                n.to_string()
            }
        }
        JsonValue::String(s) => py_str(s),
        JsonValue::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_json).collect();
            format!("[{}]", parts.join(", "))
        }
        JsonValue::Object(map) => {
            // serde_json preserves insertion order (its "preserve_order"
            // feature is on by default); Python's json.loads does too.
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_str(k), py_json(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python `repr()` of a JSON value (as parsed by serde_json).
/// This is what `ollada ps` prints: `print(json.loads(resp.read()))`.
pub fn py_json(value: &JsonValue) -> String {
    py_json_scalar(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_match_python_repr() {
        assert_eq!(py_float(16.0), "16.0");
        assert_eq!(py_float(7.66), "7.66");
        assert_eq!(py_float(4.1), "4.1");
        assert_eq!(py_float(6.59), "6.59");
        assert_eq!(py_float(0.0), "0.0");
        assert_eq!(py_float(0.9), "0.9");
        assert_eq!(py_float(1.0), "1.0");
    }

    #[test]
    fn strings_use_single_quotes_by_default() {
        assert_eq!(py_str("hello"), "'hello'");
        assert_eq!(py_str(""), "''");
    }

    #[test]
    fn strings_with_a_single_quote_use_double_quotes() {
        // Python: repr("ollada's own") == '"ollada\'s own"' — double quotes.
        assert_eq!(py_str("ollada's own"), "\"ollada's own\"");
    }

    #[test]
    fn strings_with_both_quotes_escape_the_single() {
        assert_eq!(py_str("it's \"quoted\""), "'it\\'s \"quoted\"'");
    }

    #[test]
    fn strings_escape_controls() {
        assert_eq!(py_str("a\nb\tc\\d"), "'a\\nb\\tc\\\\d'");
    }

    #[test]
    fn yaml_dict_matches_python_repr() {
        let value: YamlValue = serde_yaml::from_str(
            "{steps: 256, remasking: low_confidence, temperature: 0.0, threshold: null}",
        )
        .unwrap();
        assert_eq!(
            py_yaml(&value),
            "{'steps': 256, 'remasking': 'low_confidence', 'temperature': 0.0, 'threshold': None}"
        );
    }

    #[test]
    fn json_dict_matches_python_repr() {
        let value: JsonValue =
            serde_json::from_str(r#"{"loaded": null, "models": ["dream:7b"], "count": 1}"#)
                .unwrap();
        assert_eq!(
            py_json(&value),
            "{'loaded': None, 'models': ['dream:7b'], 'count': 1}"
        );
    }
}
