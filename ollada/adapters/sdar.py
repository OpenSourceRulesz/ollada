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

from ollada.adapters.base import BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_FUSED_FILENAME = "fused_linear_diffusion_cross_entropy.py"
_FUSED_GITHUB_RAW = (
    "https://raw.githubusercontent.com/JetAstra/SDAR/main/"
    "training/model/{flavor}/" + _FUSED_FILENAME
)


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


def _register_flash_attn_fallback() -> str:
    """Make `flash_attn.ops.triton.layer_norm.rms_norm_fn` importable.

    Returns "real" if flash_attn is installed, "torch-fallback" if the
    stand-in module chain was registered in sys.modules.
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
    return "torch-fallback"


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
    if not fused_path.exists():
        url = _FUSED_GITHUB_RAW.format(flavor=flavor)
        urllib.request.urlretrieve(url, fused_path)
    return str(overlay), [
        f"{_FUSED_FILENAME} missing from HF repo; authors' file restored "
        "from JetAstra/SDAR GitHub (training/model/)"
    ]


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
    """[1, Q, KV] 0/1 float mask -> flex-attention BlockMask (SDAR's
    attention accepts nothing else — see adapter docstring, repair 3)."""
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
    call, eager. Returns True if the swap was applied."""
    import importlib

    try:
        module = importlib.import_module(model.__class__.__module__)
        from torch.nn.attention.flex_attention import flex_attention

        def eager_flex_attention(query, key, value, attention_mask, **kwargs):
            return flex_attention(query, key, value, block_mask=attention_mask, **kwargs)

        module.fused_flex_attention = eager_flex_attention
        return True
    except Exception:
        return False



def _forward_logits(model, input_ids, block_mask, position_ids, cache, store_kv):
    """Drive SDAR's inner model + lm_head directly (repair 4): the outer
    forward drops past_key_values, the inner one accepts it."""
    out = model.model(
        input_ids=input_ids,
        attention_mask=block_mask,
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
            _to_block_mask(attn_mask[:, :prefill_length, :prefill_length]),
            position_ids[:, :prefill_length],
            past_key_values,
            store_kv=True,
        )

    transfer_counts = _num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        lo, hi = num_block * block_length, (num_block + 1) * block_length
        cur_x = x[:, lo:hi].clone()
        cur_attn = _to_block_mask(attn_mask[:, lo:hi, :hi])
        cur_pos = position_ids[:, lo:hi]
        for step in range(denoising_steps + 1):
            mask_index = cur_x == mask_id
            if mask_index.sum() == 0:
                _forward_logits(
                    model, cur_x, cur_attn, cur_pos, past_key_values, store_kv=True
                )
                break
            logits = _forward_logits(
                model, cur_x, cur_attn, cur_pos, past_key_values, store_kv=False
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


class SdarAdapter(BaseAdapter):
    family = "sdar"

    def load(self) -> None:
        recipe = self.recipe
        notes: list[str] = []
        flash = _register_flash_attn_fallback()
        notes.append(f"flash_attn: {flash}")
        local_dir, overlay_notes = _prepare_local_dir(recipe.source)
        notes.extend(overlay_notes)
        self.load_notes = notes

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        kwargs = {
            "trust_remote_code": True,
            # transformers>=4.57 renamed torch_dtype -> dtype (deprecation
            # warning otherwise); SDAR's env guarantees >=4.57.
            "dtype": _DTYPES.get(recipe.dtype, torch.float16),
        }
        qc = quantization_config(recipe.quant_default)
        if qc is not None:
            kwargs["quantization_config"] = qc
        if qc is not None or torch.cuda.is_available():
            # Place layers directly on the GPU during load; a CPU load
            # followed by .cuda() doubles peak host+device traffic and
            # muddies OOM diagnosis.
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)
        self.model.eval()
        notes.append(f"loaded dtype: {self.model.dtype}")
        if _decompile_flex_attention(self.model):
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
        )
        result = self._finish(out, input_ids.shape[1], started)
        # The canvas can leave mask tokens in unfinished tail blocks.
        result.text = result.text.replace(self.tokenizer.mask_token or "<|MASK|>", "").strip()
        return result
