import pytest

from ollada.recipes import RecipeError, get_recipe, load_catalog, validate_recipe

EXPECTED_SHELF = {"dream:7b", "sdar:4b", "diffcoder:8b", "sdar:1.7b"}


def test_shelf_has_exactly_the_expected_models():
    # The MVP three (T4-verified) plus sdar:1.7b, added 2026-10-23 as
    # the CPU tier's first entry (Dell-verified, see its recipe).
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
    # sdar:4b earned verification on a real T4 run, 2026-10-21 (notebook v0.1.1).
    sdar = by_name["sdar:4b"]
    assert sdar.is_verified is True
    assert "T4" in sdar.verified["hardware"]
    assert sdar.verified["vram_gb"] == 8.87
    # diffcoder:8b earned verification on a real T4 run, 2026-10-21 (notebook v0.1.1).
    diffcoder = by_name["diffcoder:8b"]
    assert diffcoder.is_verified is True
    assert diffcoder.verified["vram_gb"] == 15.15
    # sdar:1.7b earned the shelf's first CPU verification on the Dell,
    # 2026-10-23: hardware-labelled as a CPU run, and it must NOT carry
    # a VRAM figure — none was measured.
    sdar17 = by_name["sdar:1.7b"]
    assert sdar17.is_verified is True
    assert "CPU" in sdar17.verified["hardware"]
    assert "vram_gb" not in sdar17.verified


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


def test_prompt_ids_unwraps_mapping_style_batch_encoding():
    """transformers 5.x BatchEncoding is a UserDict (Mapping), not a dict —
    _prompt_ids must still unwrap it to the raw ids (DiffCoder regression)."""
    from collections import UserDict

    from ollada.adapters.base import BaseAdapter

    class FakeEncoding(UserDict):
        pass

    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
            return FakeEncoding({"input_ids": [[1, 2, 3]]})

    adapter = object.__new__(BaseAdapter)
    adapter.tokenizer = FakeTokenizer()
    assert adapter._prompt_ids([{"role": "user", "content": "hi"}]) == [[1, 2, 3]]



def test_diffcoder_mask_dtype_alignment_wrapper():
    """Repair 1: the wrapper must force the model's dtype into mask calls.

    Runs in the torch-free dev venv via minimal module stubs (the adapter
    module imports torch/transformers at module level).
    """
    import sys
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "fp16"
    torch_stub.bfloat16 = "bf16"
    torch_stub.no_grad = lambda: (lambda f: f)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.BitsAndBytesConfig = object
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
    sys.modules["torch"] = torch_stub
    sys.modules["transformers"] = transformers_stub
    try:
        sys.modules.pop("ollada.adapters.diffcoder", None)
        from ollada.adapters.diffcoder import _align_mask_dtype

        class FakeModel:
            dtype = "fp16"

            def _make_block_causal_mask(self, seq_len, block_size=2, device=None, dtype="bf16"):
                return dtype

        model = FakeModel()
        assert _align_mask_dtype(model) is True
        # generate_block passes dtype=bf16 explicitly — must be overridden:
        assert model._make_block_causal_mask(8, 4, None, dtype="bf16") == "fp16"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("ollada.adapters.diffcoder", None)
