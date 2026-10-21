"""Recipe loading and validation (schema v1).

A Recipe is the ollada equivalent of Ollama's Modelfile: everything a user
should never have to guess — source repo, quantization, dtype, dependency
pins, and generation defaults — curated and (eventually) verified by a
real-load run. See PLAN.md sections 4 and 7.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*:[a-z0-9][a-z0-9\.]*$")

REQUIRED_FIELDS = (
    "name",
    "family",
    "source",
    "license",
    "task",
    "size_gb_bf16",
    "quant_default",
    "dtype",
    "generate",
)


class RecipeError(ValueError):
    """Raised when a catalog recipe is malformed."""


@dataclass
class Recipe:
    name: str
    family: str
    source: str
    license: str
    task: str
    description: str
    size_gb_bf16: float
    quant_default: str
    dtype: str
    generate: dict
    env: dict = field(default_factory=dict)
    verified: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def is_verified(self) -> bool:
        """A model counts as verified only with a real hardware run on record."""
        return bool(self.verified.get("hardware") and self.verified.get("date"))

    def to_public_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "source": self.source,
            "license": self.license,
            "task": self.task,
            "description": self.description,
            "size_gb_bf16": self.size_gb_bf16,
            "quant_default": self.quant_default,
            "verified": self.is_verified,
        }


def validate_recipe(data: dict, path: Path | None = None) -> Recipe:
    where = f" in {path}" if path else ""
    if not isinstance(data, dict):
        raise RecipeError(f"recipe is not a mapping{where}")
    for key in REQUIRED_FIELDS:
        if key not in data:
            raise RecipeError(f"recipe missing required field '{key}'{where}")
    name = str(data["name"])
    if not _NAME_RE.match(name):
        raise RecipeError(f"invalid model name '{name}' (want e.g. 'dream:7b'){where}")
    if not isinstance(data["generate"], dict) or not data["generate"]:
        raise RecipeError(f"'generate' must be a non-empty mapping{where}")
    try:
        size = float(data["size_gb_bf16"])
    except (TypeError, ValueError):
        raise RecipeError(f"'size_gb_bf16' must be a number{where}") from None
    if size <= 0:
        raise RecipeError(f"'size_gb_bf16' must be positive{where}")
    return Recipe(
        name=name,
        family=str(data["family"]),
        source=str(data["source"]),
        license=str(data["license"]),
        task=str(data["task"]),
        description=str(data.get("description", "")),
        size_gb_bf16=size,
        quant_default=str(data["quant_default"]),
        dtype=str(data["dtype"]),
        generate=dict(data["generate"]),
        env=dict(data.get("env") or {}),
        verified=dict(data.get("verified") or {}),
        path=path,
    )


def catalog_dir() -> Path:
    return Path(__file__).parent / "catalog"


def load_catalog(directory: Path | None = None) -> list[Recipe]:
    """Load every recipe on the shelf, sorted by name. Fails loudly on duplicates."""
    directory = directory or catalog_dir()
    recipes: list[Recipe] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        recipe = validate_recipe(data, path)
        if recipe.name in seen:
            raise RecipeError(f"duplicate model name '{recipe.name}' in catalog")
        seen.add(recipe.name)
        recipes.append(recipe)
    return sorted(recipes, key=lambda r: r.name)


def get_recipe(name: str, directory: Path | None = None) -> Recipe:
    for recipe in load_catalog(directory):
        if recipe.name == name:
            return recipe
    available = ", ".join(r.name for r in load_catalog(directory)) or "(empty shelf)"
    raise RecipeError(f"model '{name}' is not on the shelf. Available: {available}")
