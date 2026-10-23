"""SDAR family adapter (JetLM SDAR-Chat models).

SDAR has no model.generate(): its reference generation is a standalone
block-diffusion loop in the SDAR repo's generate.py (JetAstra/SDAR,
Apache-2.0). The loop below is adapted from that reference implementation —
same block-causal prefill with a KV cache, then per-block denoising with
confidence-based token transfer. Sampling defaults come from the recipe.

Requires transformers>=4.57 in practice: its modeling file imports
`TransformersKwargs`, which only exists from 4.57 — the README's
">=4.52.4" is stale, and SDAR does NOT share Dream's verified 4.53.3
environment after all (found by local reproduction, 2026-10-21).

Two upstream defects this adapter repairs at load time (both reproduced
locally against transformers 4.53.3 before the fix — see git history):

1. **flash_attn is a hard requirement of SDAR's modeling file, and it is
   not installable everywhere** (no PyPI wheels; no Windows build). Its
   RMSNorm calls flash_attn's Triton `rms_norm_fn`; the pure-PyTorch
   implementation sits commented out in their source. If real flash_attn
   is importable we use it. Otherwise we register a small stand-in module
   providing only `rms_norm_fn`, implemented as the mathematically
   equivalent PyTorch RMSNorm (compute in fp32, cast back) — the same
   fallback the authors left in their file.
2. **The HF repo is missing a file its own modeling imports.**
   `fused_linear_diffusion_cross_entropy.py` exists in the authors'
   GitHub repo (training/model/<model>/) but was never uploaded to HF,
   and the imported class is never used at inference (dead import). We
   load from a local overlay directory: symlinks to the HF snapshot plus
   the authors' real file fetched from their GitHub.

3. **Its attention only speaks flex-attention BlockMask.** SDAR's attention
   unconditionally calls a `torch.compile`d flex_attention wrapper with
   the mask as `block_mask`, and no code path converts a plain tensor
   mask into a BlockMask (the `_update_causal_mask` call is commented
   out in their model forward; their own generate.py passes float tensor
   masks, which cannot run as shipped). Our loop therefore builds a real
   BlockMask per call from the block-causal bool mask. And because the
   wrapper is compiled with `fullgraph=True, mode="max-autotune"` —
   minutes of compile time and fragile on older GPUs — the adapter swaps
   the module-level wrapper for the identical eager flex_attention call
   after load.

4. **Its outer forward silently drops the KV cache.** `SDARForCausalLM.
   forward` accepts `past_key_values` but its call into the inner model
   never passes it — every call starts with a fresh cache, so the
   reference loop dies at the first decode block (mask expects the prefix
   KV, attention sees only the current block). The adapter's loop
   therefore drives the inner model (`model.model`) plus `lm_head`
   directly, which is the API that actually accepts the cache and
   `store_kv`.

5. **The two modeling revisions speak different mask dialects.**
   Repair 3 describes the post-rename revision (4B). The pre-rename
   revision (1.7B) instead takes a plain bool tensor mask: its
   attention converts the mask with `.bool()`, runs SDPA with it when
   any position is masked, and — when the mask is all-True, which is
   every per-block decode step in the loop — calls `flash_attn_func`,
   which does not exist off Linux/GPU. The adapter detects the loaded
   revision's dialect at load (`_detect_mask_contract`), hands the loop
   the mask in that dialect, and extends the repair-1 stand-in package
   with an SDPA-based `flash_attn_func` (the all-visible case is the
   only one that reaches it, and full attention is exactly what it
   computes) plus loud stubs for the varlen / bert_padding names the
   older file imports but never calls.

6. **Its fused-loss module imports triton at module level.** The
   post-rename modeling imports the overlay-restored fused-loss file
   (repair 2), which does `import triton` and decorates its training
   kernels with it. Triton ships no Windows build, and the loss class
   is never used at inference, so when real triton is absent the
   adapter registers an import stub: the decorators let the module
   load, while any actual kernel call raises loudly. (The pre-rename
   modeling does not import the fused module at all.)

CPU loads: when no CUDA device is present, the adapter loads on CPU at
bfloat16 (see _resolve_load_plan) — the recipe's float16 is a T4-safe GPU
dtype, not a CPU-viable one. This is a probe path, not a verified
configuration: no CPU run has passed the section-7 gate yet.

Which repairs were applied is recorded on the adapter as `load_notes` and
belongs in any verification record for this family.

UNVERIFIED in ollada until a real-load gate passes on a T4.
"""

from __future__ import annotations

import os
import sys
import time
import types
import urllib.request
from importlib.machinery import ModuleSpec
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache

from ollada.adapters.base import AdapterError, BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_FUSED_FILENAME = "fused_linear_diffusion_cross_entropy.py"
_FUSED_GITHUB_RAW = (
    "https://raw.githubusercontent.com/JetAstra/SDAR/main/"
    "training/model/{flavor}/" + _FUSED_FILENAME
)
_FUSED_FALLBACK_FLAVOR = "SDAR-4B-Chat"


def _fused_candidate_urls(flavor: str) -> list[str]:
    """Raw-GitHub URLs to try for the authors' fused-loss module, in order.

    Upstream's training/model/ tree only carries some flavors (4B and 8B
    as of 2026-10), so the flavor-templated URL 404s for e.g.
    SDAR-1.7B-Chat. The module is byte-identical across the flavor
    folders that do exist (same git blob SHA, verified 2026-10-23), so
    the 4B copy is a sound fallback for any flavor. Pure, so the
    torch-free tests can pin the order.
    """
    urls = [_FUSED_GITHUB_RAW.format(flavor=flavor)]
    if flavor != _FUSED_FALLBACK_FLAVOR:
        urls.append(_FUSED_GITHUB_RAW.format(flavor=_FUSED_FALLBACK_FLAVOR))
    return urls


def _torch_rms_norm_fn(x, weight, bias=None, eps=1e-6, **_kwargs):
    """PyTorch RMSNorm equivalent of flash_attn's Triton rms_norm_fn.

    Matches the fallback implementation SDAR's authors left commented out
    in modeling_sdar.py: normalize in fp32, scale, cast back to input dtype.
    """
    dtype = x.dtype
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype)


def _sdpa_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_kwargs):
    """SDPA equivalent of flash_attn's flash_attn_func for the stand-in
    package. Inputs and outputs use flash-attn's (batch, seq, heads,
    dim) layout. Only reached from the pre-rename SDAR attention's
    all-visible branch (docstring repair 5), where full attention is
    exactly what flash_attn_func(causal=False) computes.
    """
    import torch.nn.functional as F

    attn = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
        enable_gqa=True,
    )
    return attn.transpose(1, 2)


def _unavailable_flash_stub(*_args, **_kwargs):
    raise RuntimeError(
        "this flash_attn entry point is an import-only stub in ollada's "
        "stand-in package (SDAR's modeling imports it but never calls "
        "it at inference); real flash-attn is required to use it"
    )


def _register_flash_attn_fallback() -> str:
    """Make the flash_attn names SDAR's modeling imports importable.

    Returns "real" if flash_attn is installed, "torch-fallback" if the
    stand-in module chain was registered in sys.modules. The stand-in
    provides the triton rms_norm_fn (repair 1), an SDPA flash_attn_func
    (repair 5), and loud stubs for the varlen / bert_padding names the
    pre-rename modeling revision imports but never calls.
    """
    try:
        import flash_attn  # noqa: F401

        return "real"
    except ImportError:
        pass

    def _mod(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, loader=None)
        module.__path__ = []  # mark as package so submodule imports resolve
        sys.modules[name] = module
        return module

    root = _mod("flash_attn")
    ops = _mod("flash_attn.ops")
    triton_pkg = _mod("flash_attn.ops.triton")
    layer_norm = _mod("flash_attn.ops.triton.layer_norm")
    layer_norm.rms_norm_fn = _torch_rms_norm_fn
    root.ops = ops
    ops.triton = triton_pkg
    triton_pkg.layer_norm = layer_norm
    root.flash_attn_func = _sdpa_flash_attn_func
    root.flash_attn_varlen_func = _unavailable_flash_stub
    bert_padding = _mod("flash_attn.bert_padding")
    bert_padding.index_first_axis = _unavailable_flash_stub
    bert_padding.pad_input = _unavailable_flash_stub
    bert_padding.unpad_input = _unavailable_flash_stub
    root.bert_padding = bert_padding
    return "torch-fallback"


def _triton_kernel_raiser(*_args, **_kwargs):
    raise RuntimeError(
        "triton is an import-only stub in this environment (no triton "
        "build exists for this platform); the SDAR fused-loss kernels "
        "it decorates are training-only and must not be called at "
        "inference"
    )


def _triton_jit(fn=None, **_kwargs):
    if fn is None:
        return lambda _f: _triton_kernel_raiser
    return _triton_kernel_raiser


def _triton_decorator_factory(*_args, **_kwargs):
    return lambda _f: _triton_kernel_raiser


class _TritonLanguageStub:
    """Attribute-tolerant stand-in for triton.language.

    Kernel bodies reference tl.* only when a kernel is compiled or
    called, but tl.constexpr appears in signature annotations that are
    evaluated at def time — so attribute access must succeed while any
    actual use stays inert.
    """

    def __getattr__(self, _name):
        return self

    def __call__(self, *_args, **_kwargs):
        return self


def _register_triton_stub() -> str:
    """Make `import triton` survivable where no triton build exists.

    See docstring repair 6. Returns "real" if triton is importable,
    otherwise registers the stub and describes it.
    """
    try:
        import triton  # noqa: F401

        return "real"
    except ImportError:
        pass

    from importlib.machinery import ModuleSpec

    def _mod(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, loader=None)
        module.__path__ = []
        sys.modules[name] = module
        return module

    root = _mod("triton")
    root.jit = _triton_jit
    root.autotune = _triton_decorator_factory
    root.heuristics = _triton_decorator_factory
    root.Config = lambda *_a, **_k: None
    root.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    root.cdiv = lambda a, b: (a + b - 1) // b
    language = _mod("triton.language")
    language.__getattr__ = lambda _name: _TritonLanguageStub()
    root.language = language
    return "import-stub (fused-loss module is training-only; kernels raise if called)"


def _register_kwargs_alias(utils_module=None) -> str | None:
    """Alias transformers.utils.LossKwargs to TransformersKwargs if absent.

    transformers 4.57 renamed the loss-kwargs TypedDict LossKwargs ->
    TransformersKwargs (its superset successor), and 5.x kept the new
    name only. SDAR's repos ship two modeling revisions: the 4B imports
    the new name, the 1.7B still imports LossKwargs — and uses it as a
    TypedDict base class — so on transformers>=4.57 the 1.7B module dies
    at import with "cannot import name 'LossKwargs'". Registering the
    successor under the old name restores exactly the composition the
    older file was written against. Returns a note when the alias was
    applied, None when the name already exists.
    """
    if utils_module is None:
        import transformers.utils as utils_module

    if hasattr(utils_module, "LossKwargs"):
        return None
    utils_module.LossKwargs = utils_module.TransformersKwargs
    return (
        "transformers.utils.LossKwargs aliased to TransformersKwargs "
        "(pre-rename SDAR modeling revision)"
    )


def _mask_contract_from_source(source: str) -> str:
    """Pure: the mask dialect a modeling source speaks."""
    return "blockmask" if "def fused_flex_attention" in source else "tensor"


def _detect_mask_contract(model) -> str:
    """Which mask dialect the loaded SDAR modeling revision speaks.

    "blockmask": the revision defines a module-level fused_flex_attention
    wrapper and its attention accepts only a flex BlockMask (the
    post-rename revision, SDAR-4B). "tensor": no such wrapper — the
    attention consumes a bool tensor mask directly (the pre-rename
    revision, SDAR-1.7B). See docstring repair 5.

    The dialect is read from the modeling module's SOURCE, never its
    live namespace: _decompile_flex_attention assigns
    module.fused_flex_attention at load, so a namespace check made
    after that step finds the adapter's own assignment and reports
    "blockmask" for every revision (the Dell run #4 bug, 2026-10-23).
    """
    module = sys.modules.get(type(model).__module__)
    source = ""
    if module is not None:
        try:
            with open(module.__file__, encoding="utf-8") as fh:
                source = fh.read()
        except (AttributeError, OSError):
            source = ""
    return _mask_contract_from_source(source)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    import shutil

    shutil.copy2(src, dst)


def _prepare_local_dir(source: str) -> tuple[str, list[str]]:
    """Return a local directory transformers can load SDAR from.

    Normally this is just the HF snapshot. If the snapshot lacks the fused
    loss module (upstream's missing upload), build an overlay of links to
    the snapshot plus the authors' file from their GitHub repo.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HOME

    snapshot = Path(snapshot_download(source))
    if (snapshot / _FUSED_FILENAME).exists():
        return str(snapshot), []

    flavor = source.rsplit("/", 1)[-1]  # e.g. SDAR-4B-Chat
    overlay = Path(HF_HOME) / "ollada_overlays" / flavor
    overlay.mkdir(parents=True, exist_ok=True)
    for item in snapshot.iterdir():
        _link_or_copy(item, overlay / item.name)
    fused_path = overlay / _FUSED_FILENAME
    if fused_path.exists() and fused_path.stat().st_size == 0:
        fused_path.unlink()  # a failed earlier fetch can leave an empty file
    restored_from = None
    if not fused_path.exists():
        errors = []
        for url in _fused_candidate_urls(flavor):
            try:
                urllib.request.urlretrieve(url, fused_path)
                restored_from = url
                break
            except Exception as exc:  # HTTPError, URLError, ...
                errors.append(f"{url}: {exc}")
        if restored_from is None:
            raise AdapterError(
                f"could not restore {_FUSED_FILENAME} for {flavor} from "
                "the authors' GitHub repo; tried: " + "; ".join(errors)
            )
    note = (
        f"{_FUSED_FILENAME} missing from HF repo; authors' file restored "
        "from JetAstra/SDAR GitHub (training/model/)"
    )
    if restored_from is not None and f"/{flavor}/" not in restored_from:
        note += (
            f" via the {_FUSED_FALLBACK_FLAVOR} copy "
            "(byte-identical across flavors)"
        )
    return str(overlay), [note]


def _top_k_logits(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    logits = logits.reshape(-1, vocab)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, 1)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def _num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    counts = torch.zeros(steps, dtype=torch.int64) + base
    counts[:remainder] += 1
    return counts


def _to_block_mask(mask3d: torch.Tensor):
    """[1, Q, KV] 0/1 float mask -> flex-attention BlockMask (the
    post-rename revision's attention accepts nothing else; the
    pre-rename revision takes the bool tensor itself — repairs 3, 5)."""
    from torch.nn.attention.flex_attention import create_block_mask

    m = mask3d[0] > 0
    return create_block_mask(
        lambda b, h, q_idx, kv_idx: m[q_idx, kv_idx],
        B=1,
        H=None,
        Q_LEN=m.shape[0],
        KV_LEN=m.shape[1],
        device=m.device,
    )


def _decompile_flex_attention(model) -> bool:
    """Swap SDAR's torch.compile(max-autotune) flex wrapper for the same
    call, eager. Returns True when the swap was applied.

    Raises AdapterError if the swap fails. The failure must be loud:
    silently keeping the compiled wrapper (the pre-2026-10-23 behavior)
    means Inductor spends minutes compiling per input shape — or fails
    mid-generation — and on CPU that turns a detectable load problem
    into an apparent hang.
    """
    import importlib

    try:
        module = importlib.import_module(model.__class__.__module__)
        from torch.nn.attention.flex_attention import flex_attention

        def eager_flex_attention(query, key, value, attention_mask, **kwargs):
            return flex_attention(query, key, value, block_mask=attention_mask, **kwargs)

        module.fused_flex_attention = eager_flex_attention
    except Exception as exc:
        raise AdapterError(
            "SDAR flex-attention decompile failed: could not swap the "
            "torch.compile(fullgraph=True, mode='max-autotune') wrapper "
            "for the eager flex_attention call. Loading anyway would "
            "leave the compiled wrapper in place — minutes of Inductor "
            "compile per shape, or a mid-generation failure (on CPU: "
            "an apparent hang)."
        ) from exc
    return True



def _forward_logits(model, input_ids, mask, position_ids, cache, store_kv, mask_contract="blockmask"):
    """Drive SDAR's inner model + lm_head directly (repair 4): the outer
    forward drops past_key_values, the inner one accepts it. The mask
    arrives as the loop's bool tensor and is converted to a flex
    BlockMask only for the revision whose attention requires one
    (repair 5)."""
    attention_mask = _to_block_mask(mask) if mask_contract == "blockmask" else mask
    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        store_kv=store_kv,
    )
    return model.lm_head(out.last_hidden_state)


@torch.no_grad()
def _block_diffusion_generate(
    model,
    input_ids,
    mask_id,
    gen_length=256,
    block_length=4,
    denoising_steps=4,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy="low_confidence_dynamic",
    confidence_threshold=0.9,
    stopping_ids=None,
    mask_contract="blockmask",
):
    """Adapted from JetAstra/SDAR generate.py (Apache-2.0)."""
    model.eval()
    device = model.device
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        _forward_logits(
            model,
            x[:, :prefill_length],
            attn_mask[:, :prefill_length, :prefill_length],
            position_ids[:, :prefill_length],
            past_key_values,
            store_kv=True,
            mask_contract=mask_contract,
        )

    transfer_counts = _num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        lo, hi = num_block * block_length, (num_block + 1) * block_length
        cur_x = x[:, lo:hi].clone()
        cur_attn = attn_mask[:, lo:hi, :hi]
        cur_pos = position_ids[:, lo:hi]
        for step in range(denoising_steps + 1):
            mask_index = cur_x == mask_id
            if mask_index.sum() == 0:
                _forward_logits(
                    model,
                    cur_x,
                    cur_attn,
                    cur_pos,
                    past_key_values,
                    store_kv=True,
                    mask_contract=mask_contract,
                )
                break
            logits = _forward_logits(
                model,
                cur_x,
                cur_attn,
                cur_pos,
                past_key_values,
                store_kv=False,
                mask_contract=mask_contract,
            )
            x0, x0_p = _sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)

            if remasking_strategy == "low_confidence_dynamic":
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf = confidence[j] > confidence_threshold
                    if high_conf.sum() >= transfer_counts[step]:
                        transfer_index[j] = high_conf
                    else:
                        _, idx = torch.topk(confidence[j], transfer_counts[step])
                        transfer_index[j, idx] = True
            elif remasking_strategy == "low_confidence_static":
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], transfer_counts[step])
                    transfer_index[j, idx] = True
            elif remasking_strategy == "sequential":
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    first = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                    transfer_index[j, first : first + transfer_counts[step]] = True
            else:
                raise ValueError(f"unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, lo:hi] = cur_x
        if stopping_ids and any(s in x[:, prompt_length:] for s in stopping_ids):
            break

    return x


def _resolve_load_plan(cuda_available: bool, recipe_dtype: str, quantized: bool) -> dict:
    """Resolve the from_pretrained plan: which dtype, which placement.

    Pure (no torch references) so the torch-free test suite can pin it.

    With CUDA available — and for quantized (bitsandbytes) loads on any
    machine — the plan is exactly the adapter's original behavior: the
    recipe dtype is honored and layers are placed with
    device_map="auto".

    Without CUDA and without quantization the model loads on CPU, and
    the dtype becomes bfloat16: PyTorch CPU fp16 matmul is slow and
    unevenly supported (norm/softmax ops upcast or error by version),
    while bf16 is the CPU-viable compute dtype. No device_map — the
    model simply stays on CPU.
    """
    if cuda_available or quantized:
        return {"dtype": recipe_dtype, "device_map": "auto", "cpu": False}
    return {"dtype": "bfloat16", "device_map": None, "cpu": True}


class SdarAdapter(BaseAdapter):
    family = "sdar"

    def load(self) -> None:
        recipe = self.recipe
        notes: list[str] = []
        flash = _register_flash_attn_fallback()
        notes.append(f"flash_attn: {flash}")
        alias_note = _register_kwargs_alias()
        if alias_note is not None:
            notes.append(alias_note)
        notes.append(f"triton: {_register_triton_stub()}")
        local_dir, overlay_notes = _prepare_local_dir(recipe.source)
        notes.extend(overlay_notes)
        self.load_notes = notes

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        qc = quantization_config(recipe.quant_default)
        plan = _resolve_load_plan(torch.cuda.is_available(), recipe.dtype, qc is not None)
        kwargs = {
            "trust_remote_code": True,
            # transformers>=4.57 renamed torch_dtype -> dtype (deprecation
            # warning otherwise); SDAR's env guarantees >=4.57.
            "dtype": _DTYPES.get(plan["dtype"], torch.float16),
        }
        if qc is not None:
            kwargs["quantization_config"] = qc
        if plan["device_map"] is not None:
            # Place layers directly on the GPU during load; a CPU load
            # followed by .cuda() doubles peak host+device traffic and
            # muddies OOM diagnosis. (The CPU plan has no device_map at
            # all — the model stays on CPU.)
            kwargs["device_map"] = plan["device_map"]
        if plan["cpu"]:
            notes.append(
                "cpu load: no CUDA — dtype forced to bfloat16 (recipe "
                f"dtype {recipe.dtype} is not CPU-viable); no device_map, "
                "model stays on CPU"
            )
        self.model = AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)
        self.model.eval()
        notes.append(f"loaded dtype: {self.model.dtype}")
        self._mask_contract = _detect_mask_contract(self.model)
        notes.append(f"mask contract: {self._mask_contract}")
        if self._mask_contract == "blockmask" and _decompile_flex_attention(self.model):
            self.load_notes.append("flex attention: eager (compile wrapper swapped out)")
        gen_cfg = GenerationConfig.from_pretrained(local_dir)
        eos = gen_cfg.eos_token_id
        self._stopping_ids = [eos] if isinstance(eos, int) else list(eos or [])
        self._mask_id = self.tokenizer(self.tokenizer.mask_token)["input_ids"][0]

    def chat(self, messages, overrides=None):
        self._require_loaded()
        params = self._params(overrides)
        input_ids = self._prompt_ids(messages).to(self.model.device)
        started = time.time()
        out = _block_diffusion_generate(
            self.model,
            input_ids,
            mask_id=self._mask_id,
            gen_length=int(params.get("gen_length", 256)),
            block_length=int(params.get("block_length", 4)),
            denoising_steps=int(params.get("denoising_steps", 4)),
            temperature=float(params.get("temperature", 1.0)),
            top_k=int(params.get("top_k", 0)),
            top_p=float(params.get("top_p", 1.0)),
            remasking_strategy=str(params.get("remasking", "low_confidence_dynamic")),
            confidence_threshold=float(params.get("confidence_threshold", 0.9)),
            stopping_ids=self._stopping_ids,
            mask_contract=self._mask_contract,
        )
        result = self._finish(out, input_ids.shape[1], started)
        # The canvas can leave mask tokens in unfinished tail blocks.
        result.text = result.text.replace(self.tokenizer.mask_token or "<|MASK|>", "").strip()
        return result
