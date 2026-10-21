import pytest

from ollada.recipes import RecipeError, get_recipe, load_catalog, validate_recipe

EXPECTED_SHELF = {"dream:7b", "nemotron:3b", "diffcoder:8b"}


def test_shelf_has_exactly_the_three_mvp_models():
    names = {r.name for r in load_catalog()}
    assert names == EXPECTED_SHELF


def test_every_recipe_is_wellformed_and_unverified_until_a_real_run():
    for recipe in load_catalog():
        assert recipe.source.count("/") == 1  # owner/model on HF
        assert recipe.size_gb_bf16 > 0
        assert recipe.generate  # non-empty defaults
        # Nothing may claim verification before a real-load run fills it in.
        assert recipe.is_verified is False


def test_get_recipe_roundtrip_and_unknown_model():
    assert get_recipe("dream:7b").family == "dream"
    with pytest.raises(RecipeError):
        get_recipe("nope:1b")


def test_validate_rejects_missing_fields_and_bad_names():
    with pytest.raises(RecipeError):
        validate_recipe({"name": "dream:7b"})
    bad = {
        "name": "No Colon",
        "family": "dream",
        "source": "x/y",
        "license": "MIT",
        "task": "chat",
        "size_gb_bf16": 1.0,
        "quant_default": "int8",
        "dtype": "float16",
        "generate": {"steps": 1},
    }
    with pytest.raises(RecipeError):
        validate_recipe(bad)
