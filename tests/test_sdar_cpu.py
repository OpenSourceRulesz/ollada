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
- The mask contract is read from the loaded modeling module's SOURCE
  (a module-level fused_flex_attention definition means the BlockMask
  dialect; its absence means the bool-tensor dialect) — never from the
  live namespace, which the flex decompile itself mutates.
- The flash_attn stand-in also registers an SDPA flash_attn_func and
  loud stubs for the varlen / bert_padding names the pre-rename
  modeling revision imports but never calls.
- When real triton is absent, an import stub lets the fused-loss
  module (which imports triton at module level) load; its decorated
  kernels raise loudly if ever called. The stub is removed from
  sys.modules when the load window closes, so later capability
  probes see triton as genuinely absent.
- During a triton-less load, torch.compile is neutralized to an
  identity decorator (restored afterwards, even on error), because
  the modeling's @torch.compile decorator would otherwise resolve
  Inductor's backend against the triton stub at import time.
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

    # transformers.utils.logging with the real module's presentation
    # API shape (get/set verbosity, progress-bar switch), stateful so
    # tests can watch _quiet_load toggle and restore it.
    utils_stub = types.ModuleType("transformers.utils")
    hf_logging_stub = types.ModuleType("transformers.utils.logging")
    hf_state = {"verbosity": 30, "bars": True, "calls": []}

    def _get_verbosity():
        return hf_state["verbosity"]

    def _set_verbosity(value):
        hf_state["verbosity"] = value
        hf_state["calls"].append(("set_verbosity", value))

    def _set_verbosity_error():
        hf_state["verbosity"] = 40
        hf_state["calls"].append(("set_verbosity_error",))

    def _is_progress_bar_enabled():
        return hf_state["bars"]

    def _disable_progress_bar():
        hf_state["bars"] = False
        hf_state["calls"].append(("disable_progress_bar",))

    def _enable_progress_bar():
        hf_state["bars"] = True
        hf_state["calls"].append(("enable_progress_bar",))

    hf_logging_stub.get_verbosity = _get_verbosity
    hf_logging_stub.set_verbosity = _set_verbosity
    hf_logging_stub.set_verbosity_error = _set_verbosity_error
    hf_logging_stub.is_progress_bar_enabled = _is_progress_bar_enabled
    hf_logging_stub.disable_progress_bar = _disable_progress_bar
    hf_logging_stub.enable_progress_bar = _enable_progress_bar
    hf_logging_stub._state = hf_state
    utils_stub.logging = hf_logging_stub
    transformers_stub.utils = utils_stub

    stubs = {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "torch.nn.functional": functional_stub,
        "torch.nn.attention": attention_stub,
        "torch.nn.attention.flex_attention": flex_stub,
        "transformers": transformers_stub,
        "transformers.cache_utils": cache_utils_stub,
        "transformers.utils": utils_stub,
        "transformers.utils.logging": hf_logging_stub,
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


def test_mask_contract_detection_reads_source_not_namespace(tmp_path):
    with _stubbed_sdar() as sdar:
        # Regression pin for Dell run #4: _decompile_flex_attention
        # ASSIGNS module.fused_flex_attention at load, so a namespace
        # check made afterwards reports "blockmask" for every revision.
        # Detection must read the modeling source. The tensor fake
        # below deliberately carries the poisoned attribute.
        blockmask_src = tmp_path / "modeling_blockmask.py"
        blockmask_src.write_text(
            "def fused_flex_attention(q, k, v, attention_mask):\n    pass\n"
        )
        tensor_src = tmp_path / "modeling_tensor.py"
        tensor_src.write_text("class SDARAttention:\n    pass\n")
        names = ["fake_modeling_blockmask", "fake_modeling_tensor"]
        try:
            for name, path, poison in (
                ("fake_modeling_blockmask", blockmask_src, False),
                ("fake_modeling_tensor", tensor_src, True),
            ):
                module = types.ModuleType(name)
                module.__file__ = str(path)
                if poison:
                    module.fused_flex_attention = lambda *a, **k: None
                sys.modules[name] = module
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_blockmask"))
                == "blockmask"
            )
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_tensor"))
                == "tensor"
            )
            # A module that cannot be found at all also falls to tensor:
            # the wrapper's definition is the only BlockMask evidence.
            assert (
                sdar._detect_mask_contract(_fake_model("fake_modeling_missing"))
                == "tensor"
            )
        finally:
            for name in names:
                sys.modules.pop(name, None)


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


def test_triton_stub_registered_when_triton_absent():
    with _stubbed_sdar() as sdar:
        keys = ["triton", "triton.language"]
        saved = {k: sys.modules.get(k) for k in keys}
        try:
            status = sdar._register_triton_stub()
            if status == "real":
                pytest.skip("real triton installed")
            assert status.startswith("import-stub")
            triton_mod = sys.modules["triton"]
            # The import-time surface the fused-loss module needs:
            # Config is called in a list comprehension, the arithmetic
            # helpers are plain functions, and the decorators wrap the
            # training kernels.
            assert triton_mod.Config({}, num_warps=4) is None
            assert triton_mod.next_power_of_2(5) == 8
            assert triton_mod.cdiv(7, 2) == 4
            with pytest.raises(RuntimeError):
                triton_mod.jit(lambda: None)()
            with pytest.raises(RuntimeError):
                triton_mod.autotune(configs=[], key=[])(lambda: None)()
            with pytest.raises(RuntimeError):
                triton_mod.heuristics({})(lambda: None)()
            # triton.language must tolerate annotation-time access
            # (tl.constexpr is evaluated when kernel defs execute).
            assert sys.modules["triton.language"].constexpr is not None
            # The stub's lifetime is the load window: after unregister,
            # the process sees triton as genuinely absent again (what
            # framework capability probes expect), while a module that
            # imported the stub during the window keeps its reference
            # — and its kernels still raise if ever called.
            held_reference = sys.modules["triton"]
            sdar._unregister_triton_stub()
            assert "triton" not in sys.modules
            assert "triton.language" not in sys.modules
            with pytest.raises(RuntimeError):
                held_reference.jit(lambda: None)()
            sdar._unregister_triton_stub()  # idempotent
        finally:
            sdar._unregister_triton_stub()
            for key, previous in saved.items():
                if previous is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = previous


def test_compile_neutralized_identity_and_restore():
    with _stubbed_sdar() as sdar:
        original = object()
        fake_torch = types.SimpleNamespace(compile=original)

        def fn():
            return "ran"

        with sdar._compile_neutralized(fake_torch):
            assert fake_torch.compile is not original
            # The modeling's usage: @torch.compile(fullgraph=True, ...)
            assert fake_torch.compile(fullgraph=True, mode="x")(fn) is fn
            # Bare form: @torch.compile
            assert fake_torch.compile(fn) is fn
        assert fake_torch.compile is original
        # Restoration holds when the window raises.
        with pytest.raises(ValueError):
            with sdar._compile_neutralized(fake_torch):
                raise ValueError("boom")
        assert fake_torch.compile is original


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


def test_no_progress_bars_tolerates_missing_hub():
    with _stubbed_sdar() as sdar:
        # The stub environment has no huggingface_hub: the helper must
        # degrade to a no-op, never break a load.
        with sdar._no_progress_bars():
            pass


def test_quiet_load_swallows_stdout_prints(capsys):
    """The store_kv scolding is a bare print() in transformers'
    auto_docstring (wheel source), reachable by neither logging nor
    warnings filters — the load window must swallow stdout itself,
    and stdout must work again afterwards."""
    with _stubbed_sdar() as sdar:
        with sdar._quiet_load():
            print("🚨 `store_kv` is part of SDARModel.forward's signature")
        print("after the window")
    out = capsys.readouterr().out
    assert "store_kv" not in out
    assert "after the window" in out


def test_quiet_load_toggles_and_restores_transformers_presentation():
    with _stubbed_sdar() as sdar:
        state = sys.modules["transformers.utils.logging"]._state
        with sdar._quiet_load():
            assert state["bars"] is False
            assert state["verbosity"] == 40
        assert state["bars"] is True
        assert state["verbosity"] == 30
        names = [c[0] for c in state["calls"]]
        assert names.index("disable_progress_bar") < names.index(
            "enable_progress_bar"
        )


def test_silence_eager_flex_warnings():
    import warnings

    with _stubbed_sdar() as sdar:
        with warnings.catch_warnings():
            sdar._silence_eager_flex_warnings()
            with warnings.catch_warnings(record=True) as seen:
                warnings.warn(
                    "flex_attention called without torch.compile() - "
                    "this will use an unfused implementation",
                    UserWarning,
                )
                warnings.warn(
                    "return_lse is deprecated and will be removed in v2.10",
                    FutureWarning,
                )
                warnings.warn("an unrelated warning", UserWarning)
        messages = [str(w.message) for w in seen]
        assert "an unrelated warning" in messages
        assert not any("flex_attention" in m for m in messages)
        assert not any("return_lse" in m for m in messages)


def test_no_progress_bars_uses_hubs_function_pair_api():
    """Regression for the first live failure of the quiet-run build:
    huggingface_hub's disable_progress_bars is a plain function
    returning None (wheel source, 0.36.2), paired with
    enable_progress_bars — not a context manager. A fake hub with
    that exact shape must round-trip: disabled inside the block,
    re-enabled after, and no TypeError from the with statement."""
    calls = []
    state = {"disabled": False}
    hub = types.ModuleType("huggingface_hub")
    utils = types.ModuleType("huggingface_hub.utils")

    def _disable(name=None):
        calls.append("disable")
        state["disabled"] = True
        return None

    def _enable(name=None):
        calls.append("enable")
        state["disabled"] = False
        return None

    utils.disable_progress_bars = _disable
    utils.enable_progress_bars = _enable
    hub.utils = utils
    saved = {
        k: sys.modules.get(k) for k in ("huggingface_hub", "huggingface_hub.utils")
    }
    sys.modules.update(
        {"huggingface_hub": hub, "huggingface_hub.utils": utils}
    )
    try:
        with _stubbed_sdar() as sdar:
            with sdar._no_progress_bars():
                assert state["disabled"] is True
            assert state["disabled"] is False
            assert calls == ["disable", "enable"]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
