import pytest

from ollada.recipes import RecipeError, get_recipe, load_catalog, validate_recipe

EXPECTED_SHELF = {"dream:7b", "sdar:4b", "diffcoder:8b"}


def test_shelf_has_exactly_the_three_mvp_models():
    names = {r.name for r in load_catalog()}
    assert names == EXPECTED_SHELF


def test_every_recipe_is_wellformed():
    for recipe in load_catalog():
        assert recipe.source.count("/") == 1  # owner/model on HF
        assert recipe.size_gb_bf16 > 0
        assert recipe.generate  # non-empty defaults


def test_verification_state_matches_the_record():
    by_name = {r.name: r for r in load_catalog()}
    # dream:7b earned verification on a real T4 run, 2026-10-21 (notebook v0.1.0).
    dream = by_name["dream:7b"]
    assert dream.is_verified is True
    assert "T4" in dream.verified["hardware"]
    assert dream.verified["vram_gb"] == 9.80
    # The rest of the shelf has not had a real-load run yet — no claiming.
    assert by_name["sdar:4b"].is_verified is False
    assert by_name["diffcoder:8b"].is_verified is False


def test_family_env_pins_match_their_custom_code():
    by_name = {r.name: r for r in load_catalog()}
    # DiffCoder's custom code targets transformers 5.x; its config was saved
    # with 5.3.0. It must NOT inherit the shelf's 4.53.3 pin.
    assert by_name["diffcoder:8b"].env["transformers"] == "==5.3.0"
    # Its generate_block defaults differ from the card example: the recipe
    # carries threshold explicitly as null (fixed transfer schedule).
    assert "threshold" in by_name["diffcoder:8b"].generate
    assert by_name["diffcoder:8b"].generate["threshold"] is None
    # SDAR and Dream share the 4.53.3-era environment.
    assert by_name["dream:7b"].env["transformers"] == "==4.53.3"
    assert by_name["sdar:4b"].env["transformers"] == ">=4.57"


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
