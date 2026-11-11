//! Minimal HTTP GET for `ollada ps`, std-only (no HTTP client dependency).
//!
//! The Python CLI uses urllib with a 3 s timeout, raises on any non-200
//! status, and prints `json.loads(body)` — a Python dict. This mirrors
//! that: connect to 127.0.0.1, GET the path, read the whole response,
//! de-chunk if needed, parse the JSON body.

use serde_json::Value as JsonValue;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Error type: any failure means "no daemon answering", exactly like
/// Python's broad `except Exception` in `_cmd_ps`. The message is
/// kept for debugging; the CLI never prints it.
#[derive(Debug)]
#[allow(dead_code)]
pub struct HttpError(pub String);

fn dechunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(line_end) = rest.iter().position(|&b| b == b'\n') {
        let size_line = String::from_utf8_lossy(&rest[..line_end]);
        let size_str = size_line.trim().split(';').next().unwrap_or("");
        let size = match usize::from_str_radix(size_str, 16) {
            Ok(s) => s,
            Err(_) => break,
        };
        let data_start = line_end + 1;
        if size == 0 {
            break;
        }
        if data_start + size > rest.len() {
            break;
        }
        out.extend_from_slice(&rest[data_start..data_start + size]);
        rest = &rest[data_start + size..];
        // Skip the trailing CRLF after the chunk.
        if rest.starts_with(b"\r\n") {
            rest = &rest[2..];
        } else if rest.starts_with(b"\n") {
            rest = &rest[1..];
        }
    }
    out
}

/// GET `http://127.0.0.1:{port}{path}` and parse the JSON body.
/// Any transport, status, or parse failure is an error.
pub fn get_json(port: u16, path: &str) -> Result<JsonValue, HttpError> {
    let addr = format!("127.0.0.1:{port}");
    let timeout = Duration::from_secs(3);
    let mut stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| HttpError(format!("bad address: {e}")))?,
        timeout,
    )
    .map_err(|e| HttpError(e.to_string()))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| HttpError(e.to_string()))?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| HttpError(e.to_string()))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| HttpError(e.to_string()))?;
    let text = String::from_utf8_lossy(&raw);
    let mut parts = text.splitn(2, "\r\n\r\n");
    let head = parts.next().unwrap_or("");
    let body = parts.next().unwrap_or("");
    let status_line = head.lines().next().unwrap_or("");
    let status: u32 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(HttpError(format!("HTTP status {status}")));
    }
    let body_bytes = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body.as_bytes())
    } else {
        body.as_bytes().to_vec()
    };
    serde_json::from_slice(&body_bytes).map_err(|e| HttpError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    fn serve_once(response: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response);
            }
        });
        port
    }

    #[test]
    fn get_json_reads_a_plain_response() {
        let port = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 33\r\n\r\n{\"loaded\": null, \"models\": [\"a\"]}",
        );
        let value = get_json(port, "/api/ps").unwrap();
        assert_eq!(value["loaded"], JsonValue::Null);
        assert_eq!(value["models"][0], "a");
    }

    #[test]
    fn get_json_dechunks() {
        // {"loaded": "dream:7b"} is 22 bytes = 0x16.
        let port = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n16\r\n{\"loaded\": \"dream:7b\"}\r\n0\r\n\r\n",
        );
        let value = get_json(port, "/api/ps").unwrap();
        assert_eq!(value["loaded"], "dream:7b");
    }

    #[test]
    fn get_json_fails_on_non_200_and_on_refused() {
        let port = serve_once(b"HTTP/1.1 500 Server Error\r\nContent-Length: 0\r\n\r\n");
        assert!(get_json(port, "/api/ps").is_err());
        // Nothing listening on this port (ephemeral, just-bound-and-dropped).
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(get_json(dead_port, "/api/ps").is_err());
    }
}
