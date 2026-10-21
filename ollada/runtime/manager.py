"""Model manager — one model loaded at a time, serial generation.

Phase 1 runs the adapter in-process. The isolated-worker upgrade (one
subprocess per family env, PLAN.md section 3) slots in behind this same
interface when the Nemotron adapter lands, because Dream and Nemotron have
incompatible transformers pins.
"""

from __future__ import annotations

import threading

from ollada.adapters import create_adapter
from ollada.adapters.base import BaseAdapter, GenResult
from ollada.recipes import Recipe, get_recipe


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._adapter: BaseAdapter | None = None

    @property
    def current_name(self) -> str | None:
        return self._adapter.recipe.name if self._adapter else None

    def status(self) -> dict:
        return {
            "loaded": self._adapter.recipe.name if self._adapter else None,
            "family": self._adapter.recipe.family if self._adapter else None,
        }

    def ensure_loaded(self, recipe_or_name) -> BaseAdapter:
        recipe: Recipe = (
            recipe_or_name if isinstance(recipe_or_name, Recipe) else get_recipe(recipe_or_name)
        )
        if self._adapter and self._adapter.recipe.name == recipe.name and self._adapter.loaded:
            return self._adapter
        if self._adapter:
            self._adapter.unload()
            self._adapter = None
        adapter = create_adapter(recipe)
        adapter.load()
        self._adapter = adapter
        return adapter

    def chat(self, name: str, messages: list[dict], overrides: dict | None = None) -> GenResult:
        with self._lock:  # one GPU, one generation at a time
            adapter = self.ensure_loaded(name)
            return adapter.chat(messages, overrides)

    def unload(self) -> None:
        with self._lock:
            if self._adapter:
                self._adapter.unload()
                self._adapter = None
