"""ollada daemon — Ollama-shaped + OpenAI-shaped HTTP APIs.

Port 11435 by default (11434 is Ollama's; they coexist on purpose).
Endpoints:
  GET  /health
  GET  /api/tags                  (Ollama-shaped model list)
  GET  /api/ps                    (what is loaded)
  POST /api/chat                  (Ollama-shaped)
  POST /api/generate              (Ollama-shaped, single prompt)
  POST /v1/chat/completions       (OpenAI-shaped, final text only in v0.1)
"""

from __future__ import annotations

import time

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

    @app.get("/api/tags")
    def tags():
        return {"models": [r.to_public_dict() for r in load_catalog()]}

    @app.get("/api/ps")
    def ps():
        return manager.status()

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
