"""ollada daemon — Ollama-shaped + OpenAI-shaped HTTP APIs.

Port 11435 by default (11434 is Ollama's; they coexist on purpose).
Endpoints:
  GET  /health
  GET  /api/tags                  (Ollama-shaped model list)
  GET  /api/ps                    (what is loaded)
  POST /api/chat                  (Ollama-shaped)
  POST /api/generate              (Ollama-shaped, single prompt)
  GET  /v1/models                 (OpenAI-shaped model list)
  POST /v1/chat/completions       (OpenAI-shaped, final text only in v0.1)
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

from ollada import __version__, fit
from ollada.recipes import RecipeError, load_catalog
from ollada.runtime.manager import ModelManager
from ollada.workers import WorkerEnvError

DEFAULT_PORT = 11435

try:
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        model: str
        messages: list[dict]
        options: dict | None = None

    class GenerateRequest(BaseModel):
        model: str
        prompt: str
        options: dict | None = None

    class OpenAIChatRequest(BaseModel):
        model: str
        messages: list[dict]
        options: dict | None = None

except ImportError:  # server extra not installed; create_app will raise properly
    ChatRequest = GenerateRequest = OpenAIChatRequest = None  # type: ignore


def create_app(manager: ModelManager | None = None):
    from fastapi import FastAPI, HTTPException

    manager = manager or ModelManager()
    app = FastAPI(title="ollada", version=__version__)

    # Small, additive API honesty fix: FastAPI's default error body is
    # {"detail": ...}, while Ollama-shaped clients look for {"error": ...}.
    # Return both, with the same message — no field is invented, and
    # existing clients reading "detail" keep working.
    from fastapi.responses import JSONResponse

    @app.exception_handler(HTTPException)
    async def ollama_shaped_error(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error": exc.detail},
        )

    def run_chat(model: str, messages: list[dict], options: dict | None):
        opts = dict(options or {})
        force = opts.pop("force", False) is True  # fit-check override, not a gen param
        try:
            return manager.chat(model, messages, opts or None, force=force)
        except RecipeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (fit.FitError, WorkerEnvError) as exc:
            # Refusals (won't fit, worker env missing/mismatched), not crashes.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # adapter load/generation failure
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, **manager.status()}

    @app.get("/api/version")
    def version():
        # Ollama-shaped version check. Open WebUI calls this during
        # connection verify; without it verify fails with HTTP 404
        # even though /api/tags and /api/ps succeed. Returns
        # ollada's own version, not Ollama's.
        return {"version": __version__}

    @app.get("/api/tags")
    def tags():
        return {"models": [r.to_public_dict() for r in load_catalog()]}

    @app.get("/api/ps")
    def ps():
        # Additive only: keep ollada's own {"loaded", "family"} shape
        # (ollada CLI ps prints it), and add Ollama's "models" list
        # alongside it. Entries carry only honest fields — name, model,
        # family — no invented size/digest, which recipes do not contain.
        status = manager.status()
        loaded = status.get("loaded")
        models = (
            [{"name": loaded, "model": loaded, "family": status.get("family")}]
            if loaded
            else []
        )
        return {**status, "models": models}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        result = run_chat(req.model, req.messages, req.options)
        return {
            "model": req.model,
            "message": {"role": "assistant", "content": result.text},
            "done": True,
            "eval_count": result.tokens,
            "eval_duration_s": round(result.seconds, 3),
            "nfe": result.nfe,
        }

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        result = run_chat(req.model, [{"role": "user", "content": req.prompt}], req.options)
        return {
            "model": req.model,
            "response": result.text,
            "done": True,
            "eval_count": result.tokens,
            "eval_duration_s": round(result.seconds, 3),
            "nfe": result.nfe,
        }

    @app.get("/v1/models")
    def openai_models():
        # OpenAI-shaped discovery: the route OpenAI clients call first
        # to list models. Only fields we can fill honestly are returned:
        # id is the ollada model name, created is the recipe's verified
        # date as a unix timestamp (0 when a recipe has no parseable
        # date), owned_by is the source repo's owner. There is no
        # digest or byte size in any recipe, so those fields are left
        # absent rather than invented — a client that requires them
        # must not be given a fake value.
        data = []
        for recipe in load_catalog():
            created = 0
            raw_date = (recipe.verified or {}).get("date")
            if raw_date:
                try:
                    created = int(
                        datetime.combine(
                            date.fromisoformat(str(raw_date)), datetime.min.time(), tzinfo=timezone.utc
                        ).timestamp()
                    )
                except ValueError:
                    created = 0
            data.append(
                {
                    "id": recipe.name,
                    "object": "model",
                    "created": created,
                    "owned_by": recipe.source.split("/", 1)[0] if "/" in recipe.source else "ollada",
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def openai_chat(req: OpenAIChatRequest):
        result = run_chat(req.model, req.messages, req.options)
        return {
            "id": f"chatcmpl-ollada-{int(time.time())}",
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": result.tokens},
        }

    app.state.manager = manager
    return app


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    serve()
