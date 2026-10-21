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
    if recipe.family == "sdar":
        from ollada.adapters.sdar import SdarAdapter

        return SdarAdapter(recipe)
    if recipe.family == "nemotron":
        # Dormant: Nemotron is the first planned post-v0.1 addition. Its
        # adapter stays in the tree; it joins the shelf (catalog) later,
        # together with the isolated-worker build its transformers>=5
        # requirement forces (PLAN.md section 5, decision record).
        from ollada.adapters.nemotron import NemotronAdapter

        return NemotronAdapter(recipe)
    if recipe.family == "diffcoder":
        from ollada.adapters.diffcoder import DiffCoderAdapter

        return DiffCoderAdapter(recipe)
    raise AdapterError(f"no adapter for family '{recipe.family}' (model {recipe.name})")
