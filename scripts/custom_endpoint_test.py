"""Custom-endpoint test for ollada serve (http://127.0.0.1:11435).

What this checks, and why:
  A "custom endpoint" is when another app lets you type in a base URL
  instead of using Ollama's default. That app then talks to ollada
  the way it would talk to Ollama (Ollama-shaped) or to OpenAI
  (OpenAI-shaped). This script mimics those apps, so we can say
  honestly which ones work, which partly work, and which break.

  It does NOT install Open WebUI or any other app. It sends the
  same requests those apps send, and records the real responses.

Usage:
  1. In one terminal: ollada serve   (or: python -m ollada.cli serve)
  2. In another: python scripts/custom_endpoint_test.py [--base http://127.0.0.1:11435]
     Add --with-chat to also run a real generation (slow on CPU).
     Without --with-chat it only checks list/health/shape endpoints,
     which need no model load.

Checks:
  GET  /health
  GET  /api/tags            -- Ollama list shape: name? modified_at? size? digest?
  GET  /api/ps
  GET  /v1/models           -- many OpenAI clients list here first; ollada has none (expected 404)
  POST /api/chat            -- normal, and with stream:true (Ollama/Open WebUI style)
  POST /api/generate        -- normal
  POST /v1/chat/completions -- normal, and with stream:true + temperature (OpenAI style)

Results are printed as PASS / PARTIAL / FAIL with the actual
response keys, so a gap is a fact, not a guess. Raw responses go
to the JSON file given by --out (default: custom_endpoint_results.json
in the current directory).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error


def req(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if data:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {"error": "non-json error body"}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def check(name, ok, detail, results):
    status = "PASS" if ok == "pass" else "PARTIAL" if ok == "partial" else "FAIL"
    results.append({"check": name, "result": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="sdar:1.7b")
    ap.add_argument("--with-chat", action="store_true",
                    help="Also run real generations (slow on CPU).")
    ap.add_argument("--out", default="custom_endpoint_results.json")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    results = []

    s, health = req("GET", f"{base}/health")
    check("GET /health", "pass" if s == 200 and health.get("status") == "ok" else "fail",
          f"HTTP {s}, keys={sorted(health)}", results)

    s, tags = req("GET", f"{base}/api/tags")
    models = tags.get("models", []) if isinstance(tags, dict) else []
    names = [m.get("name") for m in models if isinstance(m, dict)]
    ollama_keys = {"name", "model", "modified_at", "size", "digest"}
    first_keys = set(models[0]) if models else set()
    missing = sorted(ollama_keys - first_keys)
    check("GET /api/tags shape",
          "pass" if s == 200 and args.model in names and not missing else "partial",
          f"HTTP {s}, {len(models)} models, first keys={sorted(first_keys)}, "
          f"Ollama-style keys missing={missing or 'none'}", results)

    s, ps = req("GET", f"{base}/api/ps")
    check("GET /api/ps", "pass" if s == 200 else "fail", f"HTTP {s}", results)

    s, v1models = req("GET", f"{base}/v1/models")
    check("GET /v1/models",
          "pass" if s == 200 else "fail",
          f"HTTP {s} (OpenAI clients often list here first; 404 means they cannot list via OpenAI shape)",
          results)

    if args.with_chat:
        prompt = [{"role": "user", "content": "What is the capital of France? Answer with exactly one word."}]
        # Normal Ollama-style chat. force=True is sandbox-specific:
        # the sandbox has less RAM than sdar:1.7b's 8 GB fit floor.
        s, chat = req("POST", f"{base}/api/chat",
                      {"model": args.model, "messages": prompt, "options": {"force": True}})
        content = ""
        try:
            content = chat.get("message", {}).get("content", "")
        except Exception:
            pass
        check("POST /api/chat (normal)",
              "pass" if s == 200 and content else "fail",
              f"HTTP {s}, done={chat.get('done')}, content={content!r:.80}", results)

        # App-style: Open WebUI sends stream:true and expects NDJSON.
        # ollada has no stream field, so it returns one final JSON.
        s, chat_stream = req("POST", f"{base}/api/chat",
                             {"model": args.model, "messages": prompt,
                              "stream": True, "options": {"force": True}})
        check("POST /api/chat (stream:true)",
              "partial" if s == 200 else "fail",
              f"HTTP {s}, response is a single JSON object, not NDJSON stream "
              f"(done={chat_stream.get('done')}); a streaming client may wait or time out",
              results)

        s, gen = req("POST", f"{base}/api/generate",
                     {"model": args.model, "prompt": "What is the capital of France?",
                      "options": {"force": True}})
        check("POST /api/generate",
              "pass" if s == 200 and gen.get("response") else "fail",
              f"HTTP {s}, response={str(gen.get('response'))[:80]!r}", results)

        s, oai = req("POST", f"{base}/v1/chat/completions",
                     {"model": args.model, "messages": prompt, "options": {"force": True}})
        choice = {}
        try:
            choice = oai.get("choices", [{}])[0]
        except Exception:
            pass
        check("POST /v1/chat/completions (normal)",
              "pass" if s == 200 and choice.get("message", {}).get("content") else "fail",
              f"HTTP {s}, content={str(choice.get('message', {}).get('content'))[:80]!r}", results)

        s, oai_stream = req("POST", f"{base}/v1/chat/completions",
                            {"model": args.model, "messages": prompt,
                             "stream": True, "temperature": 0.5, "options": {"force": True}})
        check("POST /v1/chat/completions (stream:true)",
              "partial" if s == 200 else "fail",
              f"HTTP {s}, response is a single chat.completion object, not SSE stream; "
              f"OpenAI temperature/max_tokens fields are ignored (ollada reads only options)",
              results)
    else:
        print("Skipping generation checks (run with --with-chat for a real model load).")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out} ({len(results)} checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
