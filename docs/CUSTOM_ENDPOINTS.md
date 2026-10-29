# Using ollada as a custom endpoint

A "custom endpoint" is when another app lets you type in a base URL
instead of using its default model provider. With ollada, that URL is
your local daemon:

```
http://127.0.0.1:11435
```

Start it first:

```
ollada serve
```

Then point the other app at that URL. Two shapes are available,
and they are not the same:

## Ollama-shaped (best for Ollama-style apps)

- `GET /api/tags` — lists the five shelf models by short name
  (`sdar:1.7b`, `sdar:4b`, `dream:7b`, `diffcoder:8b`, `nemotron:3b`).
- `GET /api/version` — returns ollada's own version
  (`{"version": "0.3.2"}`), not Ollama's. Added 2026-10-29 after a
  real Open WebUI connection verify failed with HTTP 404 on
  this endpoint, even though `/api/tags` and `/api/ps` returned
  200.
- `POST /api/chat` — `{"model", "messages", "options"}`, returns one
  final JSON object with `message.content`, `done: true`,
  `eval_count`, and `nfe`.
- `POST /api/generate` — same, with a single `prompt` and a
  `response` field.

## OpenAI-shaped (for OpenAI-compatible apps)

- `POST /v1/chat/completions` — `{"model", "messages"}`,
  returns a `chat.completion` object with one choice and
  `usage.completion_tokens`.

## What works, what partly works, what breaks

Verified against a live `ollada serve` (0.3.2, sandbox, 2026-10-29)
with `scripts/custom_endpoint_test.py`. Run that script yourself
against your own daemon before relying on a new app:

- **Works:** `GET /health`, `GET /api/version`, `GET /api/ps`,
  `GET /api/tags` (names are correct), `POST /api/chat`,
  `POST /api/generate`, `POST /v1/chat/completions` — all as
  non-streaming calls.
- **Partly works:** `GET /api/tags` returns ollada's own fields
  (`name`, `model` — same as `name`, added 2026-10-29 after a
  real Open WebUI dropdown showed "No models available" without
  it — `family`, `source`, `license`, `task`, `description`,
  `size_gb_bf16`, `verified`). It does NOT return Ollama's
  `modified_at`, `size` (bytes), or `digest`. An app that only
  needs names will list fine; an app that requires those fields
  may show a blank size/date.
- **Breaks / not present:**
  - **No streaming — verified live, not just from code.**
    `POST /api/chat` with `stream: true` returned one final
    JSON object (`{"model": "sdar:1.7b", "message": {"content":
    "Paris"}, "done": true, "eval_count": 131,
    "eval_duration_s": 114.331}`), not NDJSON (Ollama) or SSE
    (OpenAI). The 114s here is a sandbox-CPU time, not a shelf
    speed — do not quote it as one. Open WebUI, which sends
    `stream: true` by default, still works with this: it
    displays the single final answer at once, with no
    streaming tokens, no diffusion/watch animation, and no
    streaming tok/s display (verified 2026-10-29, see below).
    An app that strictly requires a stream could still wait
    or time out, but Open WebUI does not.
  - **No `GET /v1/models`.** OpenAI-style apps that list models
    from that endpoint cannot list ollada's models that way.
    Use the Ollama-shaped `/api/tags` list, or type the model
    name in by hand.
  - **No CORS headers, and the daemon binds `127.0.0.1` only.**
    A server-side app on the same machine (for example, Open
    WebUI run locally) can reach it. A web page served from
    another origin, or a remote server, cannot — by design.
    Do not expose this daemon to the internet: it has no auth.
  - **OpenAI generation knobs are ignored.** `temperature`,
    `max_tokens`, and similar top-level OpenAI fields are not
    read; ollada reads only `options` (plus `options.force`
    for the fit-check override). Set generation knobs through
    ollada's own `options` where the adapter supports them.

## A tested local example

Open WebUI run on the *same machine* as `ollada serve`, with
its Ollama base URL set to `http://127.0.0.1:11437` (any
free local port works; use `127.0.0.1`, not `localhost`, if
your machine resolves `localhost` to IPv6 first), is the
pattern this was designed for. **Verified end-to-end
2026-10-29 on a laptop CPU** (Intel Core Ultra 7 365,
32GB RAM, Windows 11, CPU only, ollada 0.3.2 plus the
`/api/version` and `/api/tags` `model` fixes now on main):
Open WebUI installed via pip in a Python 3.12 venv, first
local account created, Ollama base URL set to the local
serve, verify passed, all five models listed, and chat
worked with `sdar:1.7b` ("The capital of France is Paris.")
and with `nemotron:3b` — plain single answers, no
streaming, as above. Two setup notes from that run:
Open WebUI needs Python `>=3.11,<3.13`, so a Python 3.14
venv cannot install it (use `uv python install 3.12` plus
`uv pip install open-webui` if you only have 3.14), and
its default OpenAI connection will error with "Missing
bearer authentication" if you have no OpenAI key — that
error is unrelated to ollada; disable that connection
and ignore it.

## Fit and load notes

- A model that will not fit is refused with HTTP 409, not
  silently loaded. Pass `options: {"force": true}` to override,
  as the CLI `--force` flag does.
- First load builds a per-family worker environment; it is
  slow the first time and fast after that.
- CPU runs are slow on long answers. That is a property of
  the models, not of the endpoint.
