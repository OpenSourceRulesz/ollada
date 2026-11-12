//! Phase 2 — the HTTP server: an axum port of `ollada/server.py`
//! (FastAPI + uvicorn), same routes, same JSON shapes, same
//! localhost-only no-auth posture. It talks to the Python workers
//! through the Rust `ModelManager` (`manager.rs`), never directly.
//!
//! Routes: `GET /health`, `GET /api/version`, `GET /api/tags`,
//! `GET /api/ps`, `POST /api/chat`, `POST /api/generate`,
//! `GET /v1/models`, `POST /v1/chat/completions`,
//! `POST /v1/completions` (prompt-only, NOT FIM: a `suffix` is
//! accepted and ignored, never honoured).
//!
//! Error mapping, exactly like Python's `run_chat`:
//! unknown model (`RecipeError`) -> 404; fit refusal or worker-env
//! error -> 409; any other adapter/load/generation failure -> 500
//! with `"<TypeName>: <message>"`. Handler-error bodies carry BOTH
//! `{"detail": ..., "error": ...}` with the same message.
//!
//! Differences from Python/FastAPI, all deliberate and documented:
//! - **Validation errors are pydantic-shaped, not pydantic-identical.**
//!   FastAPI returns 422 `{"detail": [...]}` with pydantic v2 error
//!   objects (`type` / `loc` / `msg` / `input`, plus `ctx` for JSON
//!   errors). This server builds the same shape for the same cases
//!   (missing field, wrong type, malformed JSON, non-JSON
//!   `Content-Type`, non-object body), but the `ctx.error` text for
//!   malformed JSON comes from serde_json, not Python's `json`
//!   module, so the wording differs; and pydantic coerces numeric
//!   strings for `max_tokens` / `temperature` / `top_p` (matched
//!   here) but also accepts a few exotic spellings this server
//!   rejects (e.g. surrounding whitespace in a numeric string).
//!   The status (422) and the `detail`-only body match.
//! - **Trailing slashes 307-redirect** to the bare path with an
//!   absolute `Location`, like Starlette's `redirect_slashes`.
//! - **Message entries must carry string `role` and `content`.**
//!   Python's pydantic model only checks that `messages` is a list
//!   of dicts; a dict missing `content` passes validation and fails
//!   later in the worker (500 after the model loads). Rust fails it
//!   fast with a 422 instead of loading a model to report it.
//! - **Routing 404/405 bodies carry `detail` only** (verified live
//!   against Python: `{"detail": "Not Found"}` /
//!   `{"detail": "Method Not Allowed"}` — the custom
//!   `detail`+`error` handler does not apply to routing errors).
//! - **No `server: uvicorn` header.** axum/hyper sends its own
//!   `date` header but no `server` header; the header is not part
//!   of the API shape.
//! - **No graceful shutdown hook.** Ctrl-C / SIGTERM kills the
//!   process; the manager's atexit registry still kills live
//!   workers, like Python's `_kill_all_live`.
//! - **A catalog that cannot be read** returns plain-text
//!   `Internal Server Error` (500), matching Starlette's
//!   `ServerErrorMiddleware` for an unhandled exception. (Python's
//!   `load_catalog` failure inside `/api/tags` / `/v1/models` is
//!   likewise unhandled there.)
//! - **`eval_duration_s` rounding** uses Rust's round-half-away
//!   (`(s * 1000.0).round() / 1000.0`); Python's `round()` is
//!   round-half-even. They differ only on exact .0005 boundaries.
//! - **Non-ASCII text** is emitted as raw UTF-8, not `\uXXXX`
//!   escapes (Python's `json.dumps` defaults to `ensure_ascii`).
//!   JSON-equivalent, not byte-identical, for non-ASCII payloads.
//!   (The catalog itself is ASCII, so `/api/tags` / `/v1/models`
//!   are byte-identical; only generated text could differ.)

use crate::manager::{ChatMessage, GenResult, ManagerError, ModelManager};
use crate::recipes::{load_catalog, Recipe};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// The ollada version the server reports. Source of truth:
/// `ollada/__init__.py` `__version__` (which matches
/// `pyproject.toml` `version`); the `version_matches_python_package`
/// test reads that file and fails if they drift. This is the
/// *ollada* version, matching the Rust crate's version.
pub const OLLADA_VERSION: &str = "0.3.7";

/// Default bind, matching Python's `serve()` / CLI defaults.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 11435;

/// Shared state: one `ModelManager` for the whole server (the CLI's
/// `run` path and the server share the manager instance, so a model
/// loaded by one is loaded for the other).
pub struct ServerState {
    pub manager: Arc<ModelManager>,
    pub catalog_dir: PathBuf,
}

pub fn build_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/ps", get(ps))
        .route("/api/chat", post(api_chat))
        .route("/api/generate", post(api_generate))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/completions", post(openai_completions))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(state)
}

/// Bind and serve forever. The caller prints the `ollada serving
/// on ...` line first, like Python's `_cmd_serve`.
pub async fn serve(state: Arc<ServerState>, host: &str, port: u16) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    axum::serve(listener, build_router(state)).await
}

// -- response helpers ---------------------------------------------------------

/// `{"detail": msg, "error": msg}` — the handler-error body.
fn detail_and_error(status: StatusCode, msg: &str) -> Response {
    let mut map = JsonMap::new();
    map.insert("detail".to_string(), JsonValue::String(msg.to_string()));
    map.insert("error".to_string(), JsonValue::String(msg.to_string()));
    (status, Json(JsonValue::Object(map))).into_response()
}

/// `{"detail": ...}` only — routing errors and validation errors.
fn detail_only(status: StatusCode, detail: JsonValue) -> Response {
    let mut map = JsonMap::new();
    map.insert("detail".to_string(), detail);
    (status, Json(JsonValue::Object(map))).into_response()
}

async fn not_found(uri: axum::http::Uri, headers: axum::http::HeaderMap) -> Response {
    // Starlette's redirect_slashes: a trailing slash on a known
    // route 307-redirects to the bare path (absolute Location,
    // empty body). Unknown paths stay 404.
    const ROUTES: [&str; 9] = [
        "/health",
        "/api/version",
        "/api/tags",
        "/api/ps",
        "/api/chat",
        "/api/generate",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
    ];
    let path = uri.path();
    if path.len() > 1 && path.ends_with('/') {
        let bare = &path[..path.len() - 1];
        if ROUTES.contains(&bare) {
            let host = headers
                .get(axum::http::header::HOST)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            let mut location = format!("http://{host}{bare}");
            if let Some(query) = uri.query() {
                location.push('?');
                location.push_str(query);
            }
            return (
                StatusCode::TEMPORARY_REDIRECT,
                [(axum::http::header::LOCATION, location)],
                "",
            )
                .into_response();
        }
    }
    detail_only(
        StatusCode::NOT_FOUND,
        JsonValue::String("Not Found".to_string()),
    )
}

async fn method_not_allowed() -> Response {
    detail_only(
        StatusCode::METHOD_NOT_ALLOWED,
        JsonValue::String("Method Not Allowed".to_string()),
    )
}

/// Map a manager failure the way Python's `run_chat` does.
fn map_manager_error(err: &ManagerError) -> (StatusCode, String) {
    match err {
        ManagerError::Recipe(e) => (StatusCode::NOT_FOUND, e.to_string()),
        ManagerError::Fit(e) => (StatusCode::CONFLICT, e.to_string()),
        ManagerError::Env(e) => (StatusCode::CONFLICT, e.to_string()),
        ManagerError::Worker(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("WorkerError: {e}"),
        ),
        ManagerError::Adapter(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("AdapterError: {e}"),
        ),
        // Rust-only refusal (Python loads in-process instead); there
        // is no Python type name for it, so RuntimeError is the
        // honest stand-in.
        ManagerError::InProcess(m) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("RuntimeError: {m}"),
        ),
    }
}

fn round_3(seconds: f64) -> JsonValue {
    let rounded = (seconds * 1000.0).round() / 1000.0;
    serde_json::Number::from_f64(rounded)
        .map(JsonValue::Number)
        .unwrap_or(JsonValue::Null)
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// -- GET handlers -------------------------------------------------------------

async fn health(State(state): State<Arc<ServerState>>) -> Json<JsonValue> {
    let status = state.manager.status();
    let mut map = JsonMap::new();
    map.insert("status".to_string(), JsonValue::String("ok".to_string()));
    map.insert(
        "version".to_string(),
        JsonValue::String(OLLADA_VERSION.to_string()),
    );
    map.insert(
        "loaded".to_string(),
        status
            .loaded
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    map.insert(
        "family".to_string(),
        status
            .family
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    Json(JsonValue::Object(map))
}

async fn version() -> Json<JsonValue> {
    let mut map = JsonMap::new();
    map.insert(
        "version".to_string(),
        JsonValue::String(OLLADA_VERSION.to_string()),
    );
    Json(JsonValue::Object(map))
}

async fn tags(State(state): State<Arc<ServerState>>) -> Response {
    match load_catalog(&state.catalog_dir) {
        Ok(recipes) => {
            let models: Vec<JsonValue> = recipes.iter().map(Recipe::to_public_dict).collect();
            let mut map = JsonMap::new();
            map.insert("models".to_string(), JsonValue::Array(models));
            Json(JsonValue::Object(map)).into_response()
        }
        // Unhandled in Python too: Starlette's plain-text 500.
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response(),
    }
}

async fn ps(State(state): State<Arc<ServerState>>) -> Json<JsonValue> {
    let status = state.manager.status();
    let models = match &status.loaded {
        Some(loaded) => vec![{
            let mut m = JsonMap::new();
            m.insert("name".to_string(), JsonValue::String(loaded.clone()));
            m.insert("model".to_string(), JsonValue::String(loaded.clone()));
            m.insert(
                "family".to_string(),
                status
                    .family
                    .clone()
                    .map(JsonValue::String)
                    .unwrap_or(JsonValue::Null),
            );
            JsonValue::Object(m)
        }],
        None => vec![],
    };
    let mut map = JsonMap::new();
    map.insert(
        "loaded".to_string(),
        status
            .loaded
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    map.insert(
        "family".to_string(),
        status
            .family
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    map.insert("models".to_string(), JsonValue::Array(models));
    Json(JsonValue::Object(map))
}

/// Mirrors Python's `/v1/models` `created`: the recipe's verified
/// date as a unix timestamp (midnight UTC), 0 when absent or
/// unparseable. No chrono dependency: days-from-civil arithmetic.
pub fn unix_timestamp_for_date(raw: &str) -> i64 {
    let parts: Vec<&str> = raw.split('-').collect();
    if parts.len() != 3 || parts[0].len() != 4 || parts[1].len() != 2 || parts[2].len() != 2 {
        return 0;
    }
    if !parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit())) {
        return 0;
    }
    let year: i64 = parts[0].parse().unwrap_or(0);
    let month: u32 = parts[1].parse().unwrap_or(0);
    let day: u32 = parts[2].parse().unwrap_or(0);
    if !(1..=12).contains(&month) {
        return 0;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let dim = [
        31,
        28 + u32::from(leap),
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][(month - 1) as usize];
    if day < 1 || day > dim {
        return 0;
    }
    days_from_civil(year, month, day) * 86_400
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((month + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

async fn openai_models(State(state): State<Arc<ServerState>>) -> Response {
    match load_catalog(&state.catalog_dir) {
        Ok(recipes) => {
            let mut data = Vec::new();
            for recipe in &recipes {
                let created = recipe
                    .verified
                    .get("date")
                    .and_then(|v| v.as_str())
                    .map_or(0, unix_timestamp_for_date);
                let owned_by = match recipe.source.split_once('/') {
                    Some((owner, _)) => owner.to_string(),
                    None => "ollada".to_string(),
                };
                let mut m = JsonMap::new();
                m.insert("id".to_string(), JsonValue::String(recipe.name.clone()));
                m.insert("object".to_string(), JsonValue::String("model".to_string()));
                m.insert("created".to_string(), JsonValue::from(created));
                m.insert("owned_by".to_string(), JsonValue::String(owned_by));
                data.push(JsonValue::Object(m));
            }
            let mut map = JsonMap::new();
            map.insert("object".to_string(), JsonValue::String("list".to_string()));
            map.insert("data".to_string(), JsonValue::Array(data));
            Json(JsonValue::Object(map)).into_response()
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response(),
    }
}

// -- validation (pydantic-shaped 422s) ----------------------------------------
//
// Each parser takes the raw body bytes and returns either the parsed
// request or the list of pydantic-style error objects for a
// `{"detail": [...]}` 422. Field order follows the Python request
// models, and all field errors are reported at once, like pydantic.

fn val_error(err_type: &str, loc: Vec<JsonValue>, msg: &str, input: JsonValue) -> JsonValue {
    let mut m = JsonMap::new();
    m.insert("type".to_string(), JsonValue::String(err_type.to_string()));
    m.insert("loc".to_string(), JsonValue::Array(loc));
    m.insert("msg".to_string(), JsonValue::String(msg.to_string()));
    m.insert("input".to_string(), input);
    JsonValue::Object(m)
}

fn loc_body(field: &str) -> Vec<JsonValue> {
    vec![
        JsonValue::String("body".to_string()),
        JsonValue::String(field.to_string()),
    ]
}

/// Parse the raw body as a JSON object, or return the 422 detail
/// list. Empty body -> pydantic's `missing` at `["body"]`;
/// malformed JSON -> `json_invalid`; non-object ->
/// `model_attributes_type`. When the request is not JSON
/// (`Content-Type` is not `application/json`), FastAPI does not
/// parse the body at all: pydantic receives the raw string, which
/// fails the same way (`model_attributes_type`, input = raw body).
fn parse_body_object(
    is_json: bool,
    body: &[u8],
) -> Result<JsonMap<String, JsonValue>, Vec<JsonValue>> {
    if !is_json {
        return Err(vec![val_error(
            "model_attributes_type",
            vec![JsonValue::String("body".to_string())],
            "Input should be a valid dictionary or object to extract fields from",
            JsonValue::String(String::from_utf8_lossy(body).into_owned()),
        )]);
    }
    if body.is_empty() {
        return Err(vec![val_error(
            "missing",
            vec![JsonValue::String("body".to_string())],
            "Field required",
            JsonValue::Null,
        )]);
    }
    let value: JsonValue = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            // serde_json columns are 1-based; pydantic's loc value
            // is the 0-based offset of the offending character.
            let col = e.column().saturating_sub(1);
            let mut err = val_error(
                "json_invalid",
                vec![JsonValue::String("body".to_string()), JsonValue::from(col)],
                "JSON decode error",
                JsonValue::Object(JsonMap::new()),
            );
            if let JsonValue::Object(m) = &mut err {
                let mut ctx = JsonMap::new();
                ctx.insert("error".to_string(), JsonValue::String(e.to_string()));
                m.insert("ctx".to_string(), JsonValue::Object(ctx));
            }
            return Err(vec![err]);
        }
    };
    match value {
        JsonValue::Object(map) => Ok(map),
        other => Err(vec![val_error(
            "model_attributes_type",
            vec![JsonValue::String("body".to_string())],
            "Input should be a valid dictionary or object to extract fields from",
            other,
        )]),
    }
}

/// True when the request declares a JSON body (media type
/// `application/json`, parameters such as a charset allowed).
fn is_json_content_type(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false)
}

fn req_string(
    obj: &JsonMap<String, JsonValue>,
    body: &JsonValue,
    field: &str,
    errors: &mut Vec<JsonValue>,
) -> Option<String> {
    match obj.get(field) {
        None => {
            errors.push(val_error(
                "missing",
                loc_body(field),
                "Field required",
                body.clone(),
            ));
            None
        }
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push(val_error(
                "string_type",
                loc_body(field),
                "Input should be a valid string",
                other.clone(),
            ));
            None
        }
    }
}

fn opt_string(
    obj: &JsonMap<String, JsonValue>,
    field: &str,
    errors: &mut Vec<JsonValue>,
) -> Option<String> {
    match obj.get(field) {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push(val_error(
                "string_type",
                loc_body(field),
                "Input should be a valid string",
                other.clone(),
            ));
            None
        }
    }
}

fn opt_dict(
    obj: &JsonMap<String, JsonValue>,
    field: &str,
    errors: &mut Vec<JsonValue>,
) -> Option<JsonMap<String, JsonValue>> {
    match obj.get(field) {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Object(m)) => Some(m.clone()),
        Some(other) => {
            errors.push(val_error(
                "dict_type",
                loc_body(field),
                "Input should be a valid dictionary",
                other.clone(),
            ));
            None
        }
    }
}

/// Pydantic lax mode coerces numeric strings (`"5"` -> 5) and
/// integral floats (`5.0` -> 5); a fractional float is
/// `int_from_float`, an unparseable string is `int_parsing`.
fn opt_int(
    obj: &JsonMap<String, JsonValue>,
    field: &str,
    errors: &mut Vec<JsonValue>,
) -> Option<i64> {
    match obj.get(field) {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                    Some(f as i64)
                } else {
                    errors.push(val_error(
                        "int_from_float",
                        loc_body(field),
                        "Input should be a valid integer, got a number with a fractional part",
                        JsonValue::Number(n.clone()),
                    ));
                    None
                }
            } else {
                errors.push(val_error(
                    "int_type",
                    loc_body(field),
                    "Input should be a valid integer",
                    JsonValue::Number(n.clone()),
                ));
                None
            }
        }
        Some(JsonValue::String(s)) => match s.parse::<i64>() {
            Ok(i) => Some(i),
            Err(_) => {
                errors.push(val_error(
                    "int_parsing",
                    loc_body(field),
                    "Input should be a valid integer, unable to parse string as an integer",
                    JsonValue::String(s.clone()),
                ));
                None
            }
        },
        Some(other) => {
            errors.push(val_error(
                "int_type",
                loc_body(field),
                "Input should be a valid integer",
                other.clone(),
            ));
            None
        }
    }
}

fn opt_float(
    obj: &JsonMap<String, JsonValue>,
    field: &str,
    errors: &mut Vec<JsonValue>,
) -> Option<f64> {
    match obj.get(field) {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Number(n)) => match n.as_f64() {
            Some(f) => Some(f),
            None => {
                errors.push(val_error(
                    "float_type",
                    loc_body(field),
                    "Input should be a valid number",
                    JsonValue::Number(n.clone()),
                ));
                None
            }
        },
        Some(JsonValue::String(s)) => match s.parse::<f64>() {
            Ok(f) => Some(f),
            Err(_) => {
                errors.push(val_error(
                    "float_parsing",
                    loc_body(field),
                    "Input should be a valid number, unable to parse string as a number",
                    JsonValue::String(s.clone()),
                ));
                None
            }
        },
        Some(other) => {
            errors.push(val_error(
                "float_type",
                loc_body(field),
                "Input should be a valid number",
                other.clone(),
            ));
            None
        }
    }
}

fn req_messages(
    obj: &JsonMap<String, JsonValue>,
    body: &JsonValue,
    errors: &mut Vec<JsonValue>,
) -> Option<Vec<ChatMessage>> {
    let list = match obj.get("messages") {
        None => {
            errors.push(val_error(
                "missing",
                loc_body("messages"),
                "Field required",
                body.clone(),
            ));
            return None;
        }
        Some(JsonValue::Array(a)) => a.clone(),
        Some(other) => {
            errors.push(val_error(
                "list_type",
                loc_body("messages"),
                "Input should be a valid list",
                other.clone(),
            ));
            return None;
        }
    };
    let mut messages = Vec::new();
    for (i, item) in list.iter().enumerate() {
        let entry = match item.as_object() {
            Some(m) => m,
            None => {
                let mut loc = loc_body("messages");
                loc.push(JsonValue::from(i));
                errors.push(val_error(
                    "dict_type",
                    loc,
                    "Input should be a valid dictionary",
                    item.clone(),
                ));
                continue;
            }
        };
        // Deliberate difference from Python (see module docs):
        // role/content must be present strings; Python passes a
        // malformed entry through and the worker fails with a 500
        // after loading the model.
        let mut role = None;
        let mut content = None;
        for (field, slot) in [("role", &mut role), ("content", &mut content)] {
            let mut loc = loc_body("messages");
            loc.push(JsonValue::from(i));
            loc.push(JsonValue::String(field.to_string()));
            match entry.get(field) {
                None => errors.push(val_error("missing", loc, "Field required", item.clone())),
                Some(JsonValue::String(s)) => *slot = Some(s.clone()),
                Some(other) => errors.push(val_error(
                    "string_type",
                    loc,
                    "Input should be a valid string",
                    other.clone(),
                )),
            }
        }
        if let (Some(role), Some(content)) = (role, content) {
            messages.push(ChatMessage { role, content });
        }
    }
    Some(messages)
}

#[derive(Debug)]
struct ChatBody {
    model: String,
    messages: Vec<ChatMessage>,
    options: Option<JsonMap<String, JsonValue>>,
}

#[derive(Debug)]
struct GenerateBody {
    model: String,
    prompt: String,
    options: Option<JsonMap<String, JsonValue>>,
}

#[derive(Debug)]
struct OpenAiCompletionBody {
    model: String,
    prompt: String,
    suffix: Option<String>,
    max_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    options: Option<JsonMap<String, JsonValue>>,
}

fn parse_chat_body(is_json: bool, body: &[u8]) -> Result<ChatBody, Vec<JsonValue>> {
    let obj = parse_body_object(is_json, body)?;
    let whole = JsonValue::Object(obj.clone());
    let mut errors = Vec::new();
    let model = req_string(&obj, &whole, "model", &mut errors);
    let messages = req_messages(&obj, &whole, &mut errors);
    let options = opt_dict(&obj, "options", &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(ChatBody {
        model: model.unwrap(),
        messages: messages.unwrap(),
        options,
    })
}

fn parse_generate_body(is_json: bool, body: &[u8]) -> Result<GenerateBody, Vec<JsonValue>> {
    let obj = parse_body_object(is_json, body)?;
    let whole = JsonValue::Object(obj.clone());
    let mut errors = Vec::new();
    let model = req_string(&obj, &whole, "model", &mut errors);
    let prompt = req_string(&obj, &whole, "prompt", &mut errors);
    let options = opt_dict(&obj, "options", &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(GenerateBody {
        model: model.unwrap(),
        prompt: prompt.unwrap(),
        options,
    })
}

fn parse_openai_completion_body(
    is_json: bool,
    body: &[u8],
) -> Result<OpenAiCompletionBody, Vec<JsonValue>> {
    let obj = parse_body_object(is_json, body)?;
    let whole = JsonValue::Object(obj.clone());
    let mut errors = Vec::new();
    let model = req_string(&obj, &whole, "model", &mut errors);
    let prompt = req_string(&obj, &whole, "prompt", &mut errors);
    let suffix = opt_string(&obj, "suffix", &mut errors);
    let max_tokens = opt_int(&obj, "max_tokens", &mut errors);
    let temperature = opt_float(&obj, "temperature", &mut errors);
    let top_p = opt_float(&obj, "top_p", &mut errors);
    let options = opt_dict(&obj, "options", &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(OpenAiCompletionBody {
        model: model.unwrap(),
        prompt: prompt.unwrap(),
        suffix,
        max_tokens,
        temperature,
        top_p,
        options,
    })
}

fn validation_response(errors: Vec<JsonValue>) -> Response {
    detail_only(StatusCode::UNPROCESSABLE_ENTITY, JsonValue::Array(errors))
}

// -- generation ---------------------------------------------------------------

/// Split `options` into the fit override and the generation
/// overrides, exactly like Python's `run_chat`: `force` is popped
/// (only a JSON `true` counts) and the rest is passed through, or
/// `None` when nothing is left.
fn split_options(options: Option<JsonMap<String, JsonValue>>) -> (bool, Option<JsonValue>) {
    let mut opts = options.unwrap_or_default();
    let force = opts.remove("force") == Some(JsonValue::Bool(true));
    if opts.is_empty() {
        (force, None)
    } else {
        (force, Some(JsonValue::Object(opts)))
    }
}

/// Run one generation on the manager (blocking: model load +
/// generate) and map the outcome to a response. Called inside
/// `spawn_blocking` so the tokio runtime is never stalled.
fn run_chat_blocking(
    manager: &ModelManager,
    model: &str,
    messages: &[ChatMessage],
    options: Option<JsonMap<String, JsonValue>>,
) -> Result<GenResult, (StatusCode, String)> {
    let (force, overrides) = split_options(options);
    manager
        .chat(model, messages, overrides, force, None)
        .map_err(|e| map_manager_error(&e))
}

async fn run_chat_response(
    state: &Arc<ServerState>,
    model: String,
    messages: Vec<ChatMessage>,
    options: Option<JsonMap<String, JsonValue>>,
) -> Result<GenResult, Box<Response>> {
    let manager = Arc::clone(&state.manager);
    match tokio::task::spawn_blocking(move || {
        run_chat_blocking(&manager, &model, &messages, options)
    })
    .await
    {
        Ok(Ok(result)) => Ok(result),
        Ok(Err((status, msg))) => Err(Box::new(detail_and_error(status, &msg))),
        Err(join) => Err(Box::new(detail_and_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("RuntimeError: generation task failed: {join}"),
        ))),
    }
}

async fn api_chat(
    State(state): State<Arc<ServerState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let parsed = match parse_chat_body(is_json_content_type(&headers), &body) {
        Ok(p) => p,
        Err(errors) => return validation_response(errors),
    };
    let model = parsed.model.clone();
    match run_chat_response(&state, parsed.model, parsed.messages, parsed.options).await {
        Ok(result) => {
            let mut message = JsonMap::new();
            message.insert(
                "role".to_string(),
                JsonValue::String("assistant".to_string()),
            );
            message.insert("content".to_string(), JsonValue::String(result.text));
            let mut map = JsonMap::new();
            map.insert("model".to_string(), JsonValue::String(model));
            map.insert("message".to_string(), JsonValue::Object(message));
            map.insert("done".to_string(), JsonValue::Bool(true));
            map.insert("eval_count".to_string(), JsonValue::from(result.tokens));
            map.insert("eval_duration_s".to_string(), round_3(result.seconds));
            map.insert(
                "nfe".to_string(),
                result.nfe.map(JsonValue::from).unwrap_or(JsonValue::Null),
            );
            Json(JsonValue::Object(map)).into_response()
        }
        Err(resp) => *resp,
    }
}

async fn api_generate(
    State(state): State<Arc<ServerState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let parsed = match parse_generate_body(is_json_content_type(&headers), &body) {
        Ok(p) => p,
        Err(errors) => return validation_response(errors),
    };
    let model = parsed.model.clone();
    let messages = vec![ChatMessage::user(&parsed.prompt)];
    match run_chat_response(&state, parsed.model, messages, parsed.options).await {
        Ok(result) => {
            let mut map = JsonMap::new();
            map.insert("model".to_string(), JsonValue::String(model));
            map.insert("response".to_string(), JsonValue::String(result.text));
            map.insert("done".to_string(), JsonValue::Bool(true));
            map.insert("eval_count".to_string(), JsonValue::from(result.tokens));
            map.insert("eval_duration_s".to_string(), round_3(result.seconds));
            map.insert(
                "nfe".to_string(),
                result.nfe.map(JsonValue::from).unwrap_or(JsonValue::Null),
            );
            Json(JsonValue::Object(map)).into_response()
        }
        Err(resp) => *resp,
    }
}

async fn openai_chat(
    State(state): State<Arc<ServerState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let parsed = match parse_chat_body(is_json_content_type(&headers), &body) {
        Ok(p) => p,
        Err(errors) => return validation_response(errors),
    };
    let model = parsed.model.clone();
    match run_chat_response(&state, parsed.model, parsed.messages, parsed.options).await {
        Ok(result) => {
            let mut message = JsonMap::new();
            message.insert(
                "role".to_string(),
                JsonValue::String("assistant".to_string()),
            );
            message.insert("content".to_string(), JsonValue::String(result.text));
            let mut choice = JsonMap::new();
            choice.insert("index".to_string(), JsonValue::from(0));
            choice.insert("message".to_string(), JsonValue::Object(message));
            choice.insert(
                "finish_reason".to_string(),
                JsonValue::String("stop".to_string()),
            );
            let mut usage = JsonMap::new();
            usage.insert(
                "completion_tokens".to_string(),
                JsonValue::from(result.tokens),
            );
            let mut map = JsonMap::new();
            map.insert(
                "id".to_string(),
                JsonValue::String(format!("chatcmpl-ollada-{}", unix_now_secs())),
            );
            map.insert(
                "object".to_string(),
                JsonValue::String("chat.completion".to_string()),
            );
            map.insert("model".to_string(), JsonValue::String(model));
            map.insert(
                "choices".to_string(),
                JsonValue::Array(vec![JsonValue::Object(choice)]),
            );
            map.insert("usage".to_string(), JsonValue::Object(usage));
            Json(JsonValue::Object(map)).into_response()
        }
        Err(resp) => *resp,
    }
}

async fn openai_completions(
    State(state): State<Arc<ServerState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let parsed = match parse_openai_completion_body(is_json_content_type(&headers), &body) {
        Ok(p) => p,
        Err(errors) => return validation_response(errors),
    };
    // `suffix` is accepted and deliberately IGNORED (no FIM yet).
    let _ = parsed.suffix;
    let mut options = parsed.options.unwrap_or_default();
    if parsed.max_tokens.is_some() {
        options
            .entry("gen_length".to_string())
            .or_insert_with(|| JsonValue::from(parsed.max_tokens.unwrap()));
    }
    if parsed.temperature.is_some() {
        options.entry("temperature".to_string()).or_insert_with(|| {
            serde_json::Number::from_f64(parsed.temperature.unwrap())
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null)
        });
    }
    if parsed.top_p.is_some() {
        options.entry("top_p".to_string()).or_insert_with(|| {
            serde_json::Number::from_f64(parsed.top_p.unwrap())
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null)
        });
    }
    let model = parsed.model.clone();
    let messages = vec![ChatMessage::user(&parsed.prompt)];
    match run_chat_response(&state, parsed.model, messages, Some(options)).await {
        Ok(result) => {
            let mut choice = JsonMap::new();
            choice.insert("index".to_string(), JsonValue::from(0));
            choice.insert("text".to_string(), JsonValue::String(result.text));
            choice.insert(
                "finish_reason".to_string(),
                JsonValue::String("stop".to_string()),
            );
            let mut usage = JsonMap::new();
            usage.insert(
                "completion_tokens".to_string(),
                JsonValue::from(result.tokens),
            );
            let mut map = JsonMap::new();
            map.insert(
                "id".to_string(),
                JsonValue::String(format!("cmpl-ollada-{}", unix_now_secs())),
            );
            map.insert(
                "object".to_string(),
                JsonValue::String("text_completion".to_string()),
            );
            map.insert("model".to_string(), JsonValue::String(model));
            map.insert(
                "choices".to_string(),
                JsonValue::Array(vec![JsonValue::Object(choice)]),
            );
            map.insert("usage".to_string(), JsonValue::Object(usage));
            Json(JsonValue::Object(map)).into_response()
        }
        Err(resp) => *resp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Hardware;
    use crate::manager::ModelManager;
    use crate::recipes::load_catalog as load_cat;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tower::ServiceExt;

    fn catalog_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ollada")
            .join("catalog")
    }

    fn state_with_manager(manager: ModelManager) -> Arc<ServerState> {
        Arc::new(ServerState {
            manager: Arc::new(manager),
            catalog_dir: catalog_dir(),
        })
    }

    fn plain_manager() -> ModelManager {
        let dir = std::env::temp_dir().join("ollada-rs-server-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        ModelManager::new(&catalog_dir(), &dir, &dir)
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ollada-rs-server-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn generous_hw() -> Hardware {
        Hardware {
            gpu_name: None,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: Some(64.0),
        }
    }

    fn tiny_ram_hw() -> Hardware {
        Hardware {
            gpu_name: None,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: Some(1.0),
        }
    }

    async fn response_body(resp: Response) -> (StatusCode, String) {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    async fn request(app: Router, method: Method, uri: &str, body: &str) -> (StatusCode, String) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        response_body(resp).await
    }

    // -- version ------------------------------------------------------------

    #[test]
    fn version_matches_python_package() {
        // The source of truth is ollada/__init__.py, two levels up.
        let init = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ollada")
            .join("__init__.py");
        let text = std::fs::read_to_string(&init).unwrap();
        let line = text
            .lines()
            .find(|l| l.starts_with("__version__"))
            .expect("__version__ line in ollada/__init__.py");
        let python_version = line
            .split('"')
            .nth(1)
            .expect("__version__ = \"x.y.z\" shape");
        assert_eq!(OLLADA_VERSION, python_version);
    }

    // -- public dict / timestamps --------------------------------------------

    #[test]
    fn public_dict_matches_python_shape_and_order() {
        let recipe = load_cat(&catalog_dir())
            .unwrap()
            .into_iter()
            .find(|r| r.name == "sdar:1.7b")
            .unwrap();
        let got = serde_json::to_string(&recipe.to_public_dict()).unwrap();
        assert_eq!(
            got,
            "{\"name\":\"sdar:1.7b\",\"model\":\"sdar:1.7b\",\"family\":\"sdar\",\
             \"source\":\"JetLM/SDAR-1.7B-Chat\",\"license\":\"Apache-2.0\",\"task\":\"chat\",\
             \"description\":\"SDAR 1.7B — Qwen3-1.7B converted to block diffusion; the small SDAR, and ollada's CPU pick for machines without a GPU.\",\
             \"size_gb_bf16\":4.1,\"quant_default\":\"bf16\",\"verified\":true}"
        );
    }

    #[test]
    fn created_timestamps_match_python() {
        // Values from the live Python /v1/models response.
        assert_eq!(unix_timestamp_for_date("2026-10-23"), 1_792_713_600);
        assert_eq!(unix_timestamp_for_date("2026-10-21"), 1_792_540_800);
        assert_eq!(unix_timestamp_for_date("2024-02-29"), 1_709_164_800);
        assert_eq!(unix_timestamp_for_date("2026-02-29"), 0); // not a leap year
        assert_eq!(unix_timestamp_for_date("garbage"), 0);
        assert_eq!(unix_timestamp_for_date(""), 0);
        assert_eq!(unix_timestamp_for_date("2026-13-01"), 0);
    }

    // -- error mapping ---------------------------------------------------------

    #[test]
    fn error_mapping_matches_python() {
        use crate::manager::ManagerError;
        use crate::recipes::RecipeError;
        use crate::workers::WorkerEnvError;
        let (status, msg) = map_manager_error(&ManagerError::Recipe(RecipeError(
            "model 'x' is not on the shelf".to_string(),
        )));
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(msg, "model 'x' is not on the shelf");
        let (status, _) = map_manager_error(&ManagerError::Fit(crate::manager::FitError(
            "won't fit".to_string(),
        )));
        assert_eq!(status, StatusCode::CONFLICT);
        let (status, _) =
            map_manager_error(&ManagerError::Env(WorkerEnvError("no venv".to_string())));
        assert_eq!(status, StatusCode::CONFLICT);
        let (status, msg) = map_manager_error(&ManagerError::Adapter(
            crate::manager::AdapterError("boom".to_string()),
        ));
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(msg, "AdapterError: boom");
        let (status, msg) = map_manager_error(&ManagerError::Worker(crate::manager::WorkerError(
            "died".to_string(),
        )));
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(msg, "WorkerError: died");
    }

    // -- validation ------------------------------------------------------------

    #[test]
    fn validation_missing_field_is_422_shaped() {
        let err = parse_chat_body(true, b"{\"messages\":[]}").unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0]["type"], "missing");
        assert_eq!(err[0]["loc"], serde_json::json!(["body", "model"]));
        assert_eq!(err[0]["msg"], "Field required");
    }

    #[test]
    fn validation_reports_all_field_errors_at_once() {
        let err = parse_chat_body(true, b"{}").unwrap_err();
        assert_eq!(err.len(), 2); // model and messages
    }

    #[test]
    fn validation_empty_body_and_bad_json() {
        let err = parse_chat_body(true, b"").unwrap_err();
        assert_eq!(err[0]["type"], "missing");
        assert_eq!(err[0]["loc"], serde_json::json!(["body"]));
        let err = parse_chat_body(true, b"{bad json").unwrap_err();
        assert_eq!(err[0]["type"], "json_invalid");
        assert_eq!(err[0]["loc"], serde_json::json!(["body", 1]));
        assert!(err[0].get("ctx").is_some());
    }

    #[test]
    fn validation_wrong_types() {
        let err = parse_chat_body(true, b"{\"model\":123,\"messages\":\"hi\"}").unwrap_err();
        assert_eq!(err.len(), 2);
        assert_eq!(err[0]["type"], "string_type");
        assert_eq!(err[1]["type"], "list_type");
        let err = parse_chat_body(true, b"{\"model\":\"m\",\"messages\":[\"hi\"]}").unwrap_err();
        assert_eq!(err[0]["type"], "dict_type");
        assert_eq!(err[0]["loc"], serde_json::json!(["body", "messages", 0]));
        // Message entries need string role/content (documented
        // difference: Python passes these through to the worker).
        let err = parse_chat_body(
            true,
            b"{\"model\":\"m\",\"messages\":[{\"role\":\"user\"}]}",
        )
        .unwrap_err();
        assert_eq!(err[0]["type"], "missing");
    }

    #[test]
    fn validation_numeric_coercions_match_pydantic() {
        let ok = parse_openai_completion_body(
            true,
            b"{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":\"5\",\"temperature\":\"1.5\"}",
        )
        .unwrap();
        assert_eq!(ok.max_tokens, Some(5));
        assert_eq!(ok.temperature, Some(1.5));
        let ok = parse_openai_completion_body(
            true,
            b"{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":5.0}",
        )
        .unwrap();
        assert_eq!(ok.max_tokens, Some(5));
        let err = parse_openai_completion_body(
            true,
            b"{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":5.5}",
        )
        .unwrap_err();
        assert_eq!(err[0]["type"], "int_from_float");
        let err = parse_openai_completion_body(
            true,
            b"{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":\"abc\"}",
        )
        .unwrap_err();
        assert_eq!(err[0]["type"], "int_parsing");
    }

    #[test]
    fn validation_non_json_content_type_gets_raw_string() {
        // Verified live: without application/json, FastAPI hands
        // pydantic the raw body string.
        let err = parse_chat_body(false, b"{\"model\":\"nope:1b\",\"messages\":[]}").unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0]["type"], "model_attributes_type");
        assert_eq!(err[0]["input"], "{\"model\":\"nope:1b\",\"messages\":[]}");
    }

    #[test]
    fn validation_non_object_body() {
        let err = parse_chat_body(true, b"[1,2]").unwrap_err();
        assert_eq!(err[0]["type"], "model_attributes_type");
        assert_eq!(
            err[0]["msg"],
            "Input should be a valid dictionary or object to extract fields from"
        );
    }

    #[test]
    fn split_options_pops_force_only_on_true() {
        let mut opts = JsonMap::new();
        opts.insert("force".to_string(), JsonValue::Bool(true));
        opts.insert("gen_length".to_string(), JsonValue::from(16));
        let (force, rest) = split_options(Some(opts));
        assert!(force);
        assert_eq!(rest.unwrap()["gen_length"], 16);
        let mut opts = JsonMap::new();
        opts.insert("force".to_string(), JsonValue::String("yes".to_string()));
        let (force, rest) = split_options(Some(opts));
        assert!(!force);
        assert!(rest.is_none()); // "yes" was popped, nothing left
        let (force, rest) = split_options(None);
        assert!(!force);
        assert!(rest.is_none());
    }

    // -- handlers (oneshot, no network) ----------------------------------------

    #[tokio::test]
    async fn get_endpoints_match_python_bodies() {
        let app = build_router(state_with_manager(plain_manager()));
        let (status, body) = request(app.clone(), Method::GET, "/health", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            "{\"status\":\"ok\",\"version\":\"0.3.7\",\"loaded\":null,\"family\":null}"
        );
        let (status, body) = request(app.clone(), Method::GET, "/api/version", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "{\"version\":\"0.3.7\"}");
        let (status, body) = request(app.clone(), Method::GET, "/api/ps", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "{\"loaded\":null,\"family\":null,\"models\":[]}");
        let (status, body) = request(app.clone(), Method::GET, "/v1/models", "").await;
        assert_eq!(status, StatusCode::OK);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["object"], "list");
        assert_eq!(parsed["data"].as_array().unwrap().len(), 6);
        assert_eq!(parsed["data"][0]["id"], "diffcoder:8b");
        assert_eq!(parsed["data"][0]["created"], 1_792_540_800);
        assert_eq!(parsed["data"][0]["owned_by"], "ByteDance-Seed");
        // No invented fields.
        assert!(parsed["data"][0].get("size").is_none());
        assert!(parsed["data"][0].get("digest").is_none());
        let (status, body) = request(app, Method::GET, "/api/tags", "").await;
        assert_eq!(status, StatusCode::OK);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["models"].as_array().unwrap().len(), 6);
        assert_eq!(parsed["models"][4]["name"], "sdar:1.7b");
    }

    #[tokio::test]
    async fn routing_errors_carry_detail_only() {
        let app = build_router(state_with_manager(plain_manager()));
        let (status, body) = request(app.clone(), Method::GET, "/nope", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body, "{\"detail\":\"Not Found\"}");
        let (status, body) = request(app, Method::POST, "/api/tags", "{}").await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(body, "{\"detail\":\"Method Not Allowed\"}");
    }

    #[tokio::test]
    async fn unknown_model_is_404_with_both_fields() {
        let app = build_router(state_with_manager(plain_manager()));
        let (status, body) = request(
            app,
            Method::POST,
            "/api/chat",
            "{\"model\":\"nope:1b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert!(parsed["detail"]
            .as_str()
            .unwrap()
            .contains("is not on the shelf"));
        assert_eq!(parsed["detail"], parsed["error"]);
    }

    #[tokio::test]
    async fn malformed_body_is_422_detail_only() {
        let app = build_router(state_with_manager(plain_manager()));
        let (status, body) = request(
            app,
            Method::POST,
            "/api/chat",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["detail"][0]["type"], "missing");
        assert!(parsed.get("error").is_none());
    }

    #[tokio::test]
    async fn fit_refusal_is_409_with_both_fields() {
        let manager = plain_manager();
        manager.set_hardware_override(tiny_ram_hw());
        let app = build_router(state_with_manager(manager));
        let (status, body) = request(
            app,
            Method::POST,
            "/api/chat",
            "{\"model\":\"sdar:1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert!(parsed["detail"]
            .as_str()
            .unwrap()
            .contains("host-RAM floor"));
        assert_eq!(parsed["detail"], parsed["error"]);
    }

    // Fake worker for the success path (same protocol shape as the
    // manager tests, trimmed to load + chat).
    const FAKE_WORKER: &str = r#"
import json, os, sys
print(json.dumps({"event": "ready", "pid": os.getpid()}), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError:
        continue
    rid = req.get("id")
    op = req.get("op")
    if op == "load":
        print(json.dumps({"id": rid, "ok": True, "result": {
            "loaded": req["model"], "family": "sdar",
            "supports_watch": False, "peak_gb": None, "hardware": None}}), flush=True)
    elif op == "chat":
        print(json.dumps({"id": rid, "ok": True, "result": {
            "text": "hello from fake", "tokens": 3, "seconds": 0.5,
            "nfe": None, "peak_gb": None, "hardware": None}}), flush=True)
    else:
        print(json.dumps({"id": rid, "ok": False, "error": "unknown op"}), flush=True)
"#;

    fn fake_manager(tag: &str) -> ModelManager {
        let dir = tmp_dir(tag);
        let script = dir.join("fake_worker.py");
        std::fs::write(&script, FAKE_WORKER).unwrap();
        let meas = tmp_dir(&format!("meas-{tag}"));
        let workers_dir = tmp_dir(&format!("workers-{tag}"));
        let mut manager = ModelManager::new(&catalog_dir(), &workers_dir, &meas);
        manager.set_hardware_override(generous_hw());
        manager.set_timeouts(Duration::from_secs(10), Duration::from_secs(2));
        let argv = vec![
            crate::workers::system_python()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            script.to_string_lossy().into_owned(),
        ];
        manager.set_argv_builder(Arc::new(move |_| argv.clone()));
        manager.set_env_builder(Arc::new(|_, _, _| Ok(PathBuf::from("/nonexistent/python"))));
        manager
    }

    // Test-only std mutex held across in-process oneshot awaits:
    // nothing awaited here touches the lock, so no deadlock is
    // possible; clippy cannot see that.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn chat_generate_and_openai_endpoints_use_the_manager() {
        // Serialise with the manager's own worker tests (shared
        // global live registry).
        let _guard = crate::manager::worker_test_lock();
        let app = build_router(state_with_manager(fake_manager("server-chat")));
        let (status, body) = request(
            app.clone(),
            Method::POST,
            "/api/chat",
            "{\"model\":\"sdar:1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            "{\"model\":\"sdar:1.7b\",\"message\":{\"role\":\"assistant\",\"content\":\"hello from fake\"},\"done\":true,\"eval_count\":3,\"eval_duration_s\":0.5,\"nfe\":null}"
        );
        // The model is now loaded: /api/ps shows it.
        let (status, body) = request(app.clone(), Method::GET, "/api/ps", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            "{\"loaded\":\"sdar:1.7b\",\"family\":\"sdar\",\"models\":[{\"name\":\"sdar:1.7b\",\"model\":\"sdar:1.7b\",\"family\":\"sdar\"}]}"
        );
        let (status, body) = request(
            app.clone(),
            Method::POST,
            "/api/generate",
            "{\"model\":\"sdar:1.7b\",\"prompt\":\"hi\"}",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            "{\"model\":\"sdar:1.7b\",\"response\":\"hello from fake\",\"done\":true,\"eval_count\":3,\"eval_duration_s\":0.5,\"nfe\":null}"
        );
        let (status, body) = request(
            app.clone(),
            Method::POST,
            "/v1/chat/completions",
            "{\"model\":\"sdar:1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "sdar:1.7b");
        assert!(parsed["id"]
            .as_str()
            .unwrap()
            .starts_with("chatcmpl-ollada-"));
        assert_eq!(
            parsed["choices"][0]["message"]["content"],
            "hello from fake"
        );
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["completion_tokens"], 3);
        // /v1/completions: suffix accepted and ignored, max_tokens
        // mapped to gen_length via setdefault.
        let (status, body) = request(
            app,
            Method::POST,
            "/v1/completions",
            "{\"model\":\"sdar:1.7b\",\"prompt\":\"hi\",\"suffix\":\"ignored\",\"max_tokens\":16}",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let parsed: JsonValue = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["object"], "text_completion");
        assert!(parsed["id"].as_str().unwrap().starts_with("cmpl-ollada-"));
        assert_eq!(parsed["choices"][0]["text"], "hello from fake");
        assert_eq!(parsed["usage"]["completion_tokens"], 3);
    }
}
