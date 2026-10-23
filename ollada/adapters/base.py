"""Adapter interface — one adapter per model FAMILY.

An adapter hides everything family-specific (how the model loads, what its
generate() is called, which knobs it takes) behind one interface, so the
daemon and CLI never see a family's custom code paths directly.

IMPORTANT: an adapter existing is NOT support. Support means the recipe's
verification block is filled by a real-load run on real GPU hardware
(PLAN.md section 7). Until then each adapter is unverified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ollada.recipes import Recipe


@dataclass
class GenResult:
    text: str
    tokens: int
    seconds: float
    nfe: int | None = None  # number of function evaluations, when the family reports it


class AdapterError(RuntimeError):
    pass


class BaseAdapter:
    family = "base"
    # Watch mode: an adapter that can report per-step canvas state
    # during generation sets this True and accepts on_step in chat().
    # The manager forwards a step callback only to supporting
    # adapters; everyone else is called exactly as before.
    supports_watch = False

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.model = None
        self.tokenizer = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # -- generation --------------------------------------------------------
    def chat(
        self, messages: list[dict], overrides: dict | None = None, on_step=None
    ) -> GenResult:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def _params(self, overrides: dict | None) -> dict:
        params = dict(self.recipe.generate)
        if overrides:
            params.update(overrides)
        return params

    def _prompt_ids(self, messages: list[dict]):
        """Tokenize a chat history, preferring the tokenizer's chat template."""
        from collections.abc import Mapping

        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            result = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            # BatchEncoding is a dict subclass on transformers 4.x but a
            # UserDict (Mapping, NOT a dict) on 5.x — an isinstance(dict)
            # check silently passes it through whole on 5.x, and the model's
            # generate then dies on `.shape` (DiffCoder T4 run, 2026-10-21).
            if isinstance(result, Mapping):
                result = result["input_ids"]
            return result
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
        text += "assistant:"
        return tok(text, return_tensors="pt").input_ids

    def _finish(self, out_ids, prompt_len: int, started: float, nfe: int | None = None) -> GenResult:
        new_ids = out_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenResult(
            text=text.strip(),
            tokens=int(new_ids.shape[0]),
            seconds=time.time() - started,
            nfe=nfe,
        )

    def _require_loaded(self):
        if not self.loaded:
            raise AdapterError(f"{self.recipe.name} is not loaded")


def quantization_config(quant: str):
    """Build a BitsAndBytesConfig for int8/int4, else None.

    Never pass load_in_8bit as a raw from_pretrained kwarg — it leaks into
    some families' custom __init__ and crashes (npc-engine, 2026-10-20).
    """
    if quant not in ("int8", "int4"):
        return None
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True)
