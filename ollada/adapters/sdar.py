"""SDAR family adapter (JetLM SDAR-Chat models).

SDAR has no model.generate(): its reference generation is a standalone
block-diffusion loop in the SDAR repo's generate.py (JetAstra/SDAR,
Apache-2.0). The loop below is adapted from that reference implementation —
same block-causal prefill with a KV cache, then per-block denoising with
confidence-based token transfer. Sampling defaults come from the recipe.

Requires transformers>=4.52.4 — satisfied by the shared 4.53.3 pin, so SDAR
runs in the same environment as Dream (PLAN.md section 5, decision record).

UNVERIFIED in ollada until a real-load gate passes on a T4.
"""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache

from ollada.adapters.base import BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


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
        model(
            x[:, :prefill_length],
            attention_mask=attn_mask[:, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
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
                model(
                    cur_x,
                    attention_mask=cur_attn,
                    position_ids=cur_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break
            logits = model(
                cur_x,
                attention_mask=cur_attn,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
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
        self.tokenizer = AutoTokenizer.from_pretrained(recipe.source, trust_remote_code=True)
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": _DTYPES.get(recipe.dtype, torch.float16),
        }
        qc = quantization_config(recipe.quant_default)
        if qc is not None:
            kwargs["quantization_config"] = qc
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(recipe.source, **kwargs)
        if qc is None and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        gen_cfg = GenerationConfig.from_pretrained(recipe.source)
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
