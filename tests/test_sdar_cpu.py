"""SDAR CPU load-plan tests.

The dev venv is torch-free by design (see the diffcoder wrapper test in
test_recipes.py); the SDAR adapter imports torch at module level, so
these tests register minimal module stubs just for the import, then pin
the pure load-plan helper and the flex-attention decompile contract:

- CUDA available -> the plan is exactly the pre-CPU-branch behavior
  (recipe dtype honored, device_map="auto").
- No CUDA, no quantization -> CPU plan: bfloat16, no device_map.
- A failed flex-attention decompile raises loudly instead of silently
  keeping SDAR's torch.compile(max-autotune) wrapper.
- The fused-loss restore tries the flavor's own GitHub folder first,
  then the byte-identical 4B copy (upstream carries no 1.7B folder, so
  the flavor-templated URL 404s for SDAR-1.7B-Chat).
- The mask contract is detected from the loaded modeling module:
  a module-level fused_flex_attention wrapper means the BlockMask
  dialect; its absence means the bool-tensor dialect.
- The flash_attn stand-in also registers an SDPA flash_attn_func and
  loud stubs for the varlen / bert_padding names the pre-rename
  modeling revision imports but never calls.
"""

import contextlib
import sys
import types

import pytest


@contextlib.contextmanager
def _stubbed_sdar():
    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "fp16"
    torch_stub.bfloat16 = "bf16"
    torch_stub.no_grad = lambda: (lambda f: f)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    nn_stub = types.ModuleType("torch.nn")
    functional_stub = types.ModuleType("torch.nn.functional")
    nn_stub.functional = functional_stub
    attention_stub = types.ModuleType("torch.nn.attention")
    flex_stub = types.ModuleType("torch.nn.attention.flex_attention")
    flex_stub.flex_attention = lambda *a, **k: ("flex", a, k)
    attention_stub.flex_attention = flex_stub

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.GenerationConfig = object
    transformers_stub.BitsAndBytesConfig = object
    cache_utils_stub = types.ModuleType("transformers.cache_utils")
    cache_utils_stub.DynamicCache = object

    stubs = {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "torch.nn.functional": functional_stub,
        "torch.nn.attention": attention_stub,
        "torch.nn.attention.flex_attention": flex_stub,
        "transformers": transformers_stub,
        "transformers.cache_utils": cache_utils_stub,
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("ollada.adapters.sdar", None)
    try:
        import ollada.adapters.sdar as sdar

        yield sdar
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("ollada.adapters.sdar", None)


def test_load_plan_with_cuda_is_the_original_behavior():
    with _stubbed_sdar() as sdar:
        plan = sdar._resolve_load_plan(True, "float16", quantized=False)
        assert plan["dtype"] == "float16"  # recipe dtype honored
        assert plan["device_map"] == "auto"  # layers placed on the GPU at load
        assert plan["cpu"] is False
        # Quantized CUDA loads were device_map="auto" before the branch too.
        plan_q = sdar._resolve_load_plan(True, "float16", quantized=True)
        assert plan_q["dtype"] == "float16"
        assert plan_q["device_map"] == "auto"
        assert plan_q["cpu"] is False


def test_load_plan_without_cuda_is_bf16_cpu():
    with _stubbed_sdar() as sdar:
        plan = sdar._resolve_load_plan(False, "float16", quantized=False)
        assert plan["dtype"] == "bfloat16"  # CPU fp16 is the wrong dtype
        assert plan["device_map"] is None  # model stays on CPU
        assert plan["cpu"] is True
        # The quantized path is unchanged by the branch: bitsandbytes
        # loads keep device_map="auto" and the recipe dtype either way.
        plan_q = sdar._resolve_load_plan(False, "float16", quantized=True)
        assert plan_q["dtype"] == "float16"
        assert plan_q["device_map"] == "auto"
        assert plan_q["cpu"] is False


def test_kwargs_alias_applied_when_losskwargs_missing():
    with _stubbed_sdar() as sdar:
        fake_utils = types.SimpleNamespace(TransformersKwargs="TK")
        note = sdar._register_kwargs_alias(fake_utils)
        assert fake_utils.LossKwargs == "TK"
        assert note is not None and "LossKwargs" in note


def test_kwargs_alias_noop_when_losskwargs_present():
    with _stubbed_sdar() as sdar:
        fake_utils = types.SimpleNamespace(LossKwargs="LK", TransformersKwargs="TK")
        assert sdar._register_kwargs_alias(fake_utils) is None
        assert fake_utils.LossKwargs == "LK"


def _fake_model(module_name):
    return type("FakeModel", (), {"__module__": module_name})()


def test_mask_contract_blockmask_when_fused_flex_wrapper_present():
    with _stubbed_sdar() as sdar:
        module = types.ModuleType("fake_modeling_blockmask")
        module.fused_flex_attention = lambda *a, **k: None
        sys.modules["fake_modeling_blockmask"] = module
        try:
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_blockmask"))
                == "blockmask"
            )
        finally:
            sys.modules.pop("fake_modeling_blockmask", None)


def test_mask_contract_tensor_without_fused_flex_wrapper():
    with _stubbed_sdar() as sdar:
        module = types.ModuleType("fake_modeling_tensor")
        sys.modules["fake_modeling_tensor"] = module
        try:
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_tensor"))
                == "tensor"
            )
            # A module that cannot be found at all also falls to tensor:
            # the wrapper's presence is the only BlockMask evidence.
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_missing"))
                == "tensor"
            )
        finally:
            sys.modules.pop("fake_modeling_tensor", None)


def test_flash_fallback_registers_sdpa_func_and_loud_stubs():
    with _stubbed_sdar() as sdar:
        keys = [
            "flash_attn",
            "flash_attn.ops",
            "flash_attn.ops.triton",
            "flash_attn.ops.triton.layer_norm",
            "flash_attn.bert_padding",
        ]
        saved = {k: sys.modules.get(k) for k in keys}
        try:
            if sdar._register_flash_attn_fallback() == "real":
                pytest.skip("real flash_attn installed")
            root = sys.modules["flash_attn"]
            assert callable(root.flash_attn_func)
            with pytest.raises(RuntimeError):
                root.flash_attn_varlen_func()
            with pytest.raises(RuntimeError):
                sys.modules["flash_attn.bert_padding"].pad_input()
        finally:
            for key, previous in saved.items():
                if previous is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = previous


def test_fused_candidate_urls_flavor_first_then_4b_fallback():
    with _stubbed_sdar() as sdar:
        urls = sdar._fused_candidate_urls("SDAR-1.7B-Chat")
        assert len(urls) == 2
        assert "/training/model/SDAR-1.7B-Chat/" in urls[0]
        assert "/training/model/SDAR-4B-Chat/" in urls[1]
        # The fallback flavor itself gets no duplicate entry.
        urls_4b = sdar._fused_candidate_urls("SDAR-4B-Chat")
        assert len(urls_4b) == 1
        assert "/training/model/SDAR-4B-Chat/" in urls_4b[0]


def test_decompile_failure_raises_loudly():
    with _stubbed_sdar() as sdar:
        FakeModel = type("FakeModel", (), {"__module__": "no_such_modeling_module"})
        with pytest.raises(sdar.AdapterError, match="max-autotune") as excinfo:
            sdar._decompile_flex_attention(FakeModel())
        # The original failure stays chained — loud, not swallowed.
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_decompile_success_swaps_in_the_eager_call():
    with _stubbed_sdar() as sdar:
        modeling = types.ModuleType("fake_sdar_modeling")
        modeling.fused_flex_attention = "compiled-wrapper"
        sys.modules["fake_sdar_modeling"] = modeling
        try:
            FakeModel = type("FakeModel", (), {"__module__": "fake_sdar_modeling"})
            assert sdar._decompile_flex_attention(FakeModel()) is True
            assert callable(modeling.fused_flex_attention)
            # The swap forwards the mask as block_mask=, exactly as before.
            assert modeling.fused_flex_attention(1, 2, 3, "mask") == (
                "flex",
                (1, 2, 3),
                {"block_mask": "mask"},
            )
        finally:
            sys.modules.pop("fake_sdar_modeling", None)
