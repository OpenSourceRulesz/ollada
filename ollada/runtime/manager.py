"""Model manager — one model loaded at a time, serial generation.

Phase 1 runs the adapter in-process. The isolated-worker upgrade (one
subprocess per family env, PLAN.md section 3) slots in behind this same
interface when the Nemotron adapter lands, because Dream and Nemotron have
incompatible transformers pins.
"""

from __future__ import annotations

import threading

from ollada import fit, measurements
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

    def ensure_loaded(self, recipe_or_name, force: bool = False) -> BaseAdapter:
        recipe: Recipe = (
            recipe_or_name if isinstance(recipe_or_name, Recipe) else get_recipe(recipe_or_name)
        )
        if self._adapter and self._adapter.recipe.name == recipe.name and self._adapter.loaded:
            return self._adapter
        if not force:
            # Refuse BEFORE unloading the resident model or touching the GPU:
            # a refused load leaves the current state exactly as it was.
            reason = fit.assess(recipe).blocking_reason()
            if reason:
                raise fit.FitError(reason)
        if self._adapter:
            self._adapter.unload()
            self._adapter = None
        adapter = create_adapter(recipe)
        measurements.reset_cuda_peak()
        adapter.load()
        self._adapter = adapter
        measurements.capture_and_record(recipe.name)
        return adapter

    def chat(
        self,
        name: str,
        messages: list[dict],
        overrides: dict | None = None,
        force: bool = False,
    ) -> GenResult:
        with self._lock:  # one GPU, one generation at a time
            adapter = self.ensure_loaded(name, force=force)
            result = adapter.chat(messages, overrides)
            measurements.capture_and_record(adapter.recipe.name)
            return result

    def unload(self) -> None:
        with self._lock:
            if self._adapter:
                self._adapter.unload()
                self._adapter = None
