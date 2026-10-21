"""Family adapter registry."""

from __future__ import annotations

from ollada.adapters.base import AdapterError, BaseAdapter, GenResult
from ollada.recipes import Recipe

__all__ = ["AdapterError", "BaseAdapter", "GenResult", "create_adapter"]


def create_adapter(recipe: Recipe) -> BaseAdapter:
    """Import lazily so the CLI works on machines without torch installed."""
    if recipe.family == "dream":
        from ollada.adapters.dream import DreamAdapter

        return DreamAdapter(recipe)
    if recipe.family == "nemotron":
        from ollada.adapters.nemotron import NemotronAdapter

        return NemotronAdapter(recipe)
    if recipe.family == "diffcoder":
        from ollada.adapters.diffcoder import DiffCoderAdapter

        return DiffCoderAdapter(recipe)
    raise AdapterError(f"no adapter for family '{recipe.family}' (model {recipe.name})")
