"""Stable-DiffCoder family adapter (ByteDance-Seed).

Real signature ( modeling_stable_diffcoder.py, read 2026-10-21 ):
  generate(input_ids=None, generation_config=None, **kwargs) -> plain tensor
  kwargs forward to generate_block(steps=128, gen_length=128, block_length=4,
      temperature=0.0, remasking='low_confidence', tokenizer=None,
      mask_id=5, threshold=0.95, shift=False, eos_id=None)
The model card's canonical example passes tokenizer=tokenizer,
threshold=None and eos_id=tokenizer.eos_token_id explicitly — the
generate_block defaults differ (threshold 0.95, no EOS truncation), so we
pass all three. Assertions in generate_block: gen_length % block_length == 0
and steps % (gen_length // block_length) == 0 (recipe defaults satisfy both:
256 % 4 == 0, 128 % 64 == 0). Batch size must be 1.

The model's config was saved with transformers 5.3.0 — see the recipe pin.

Load repairs applied by this adapter:
1. generate_block builds its block-causal mask with dtype=torch.bfloat16
   hardcoded, while the recipe loads fp16 on pre-Ampere GPUs (T4) —
   transformers 5.x SDPA rejects a float mask whose dtype differs from the
   query dtype (RuntimeError, T4 run 2026-10-21). The adapter wraps
   _make_block_causal_mask so the mask follows the model's dtype. The mask
   is additive (0 / -inf), so fp16 loses nothing.

UNVERIFIED in ollada until a real-load gate passes on a T4.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ollada.adapters.base import BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _align_mask_dtype(model) -> bool:
    """Wrap _make_block_causal_mask so its output dtype follows the model.

    Returns True when the wrapper was installed. See the module docstring
    (repair 1) for why this exists.
    """
    orig = getattr(model, "_make_block_causal_mask", None)
    if orig is None:
        return False
    target = model.dtype

    def patched(*args, **kwargs):
        kwargs["dtype"] = target
        return orig(*args, **kwargs)

    model._make_block_causal_mask = patched
    return True


class DiffCoderAdapter(BaseAdapter):
    family = "diffcoder"

    def load(self) -> None:
        recipe = self.recipe
        notes: list[str] = []
        self.tokenizer = AutoTokenizer.from_pretrained(recipe.source, trust_remote_code=True)
        kwargs = {
            "trust_remote_code": True,
            "dtype": _DTYPES.get(recipe.dtype, torch.float16),
        }
        qc = quantization_config(recipe.quant_default)
        if qc is not None:
            kwargs["quantization_config"] = qc
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(recipe.source, **kwargs)
        if qc is None and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        if _align_mask_dtype(self.model):
            notes.append(f"block mask dtype aligned to model dtype ({self.model.dtype})")
        self.load_notes = notes

    @torch.no_grad()
    def chat(self, messages, overrides=None):
        self._require_loaded()
        params = self._params(overrides)
        input_ids = self._prompt_ids(messages).to(self.model.device)
        started = time.time()
        threshold = params.get("threshold", None)  # card example: None (fixed schedule)
        out = self.model.generate(
            input_ids,
            steps=int(params.get("steps", 128)),
            gen_length=int(params.get("gen_length", 256)),
            block_length=int(params.get("block_length", 4)),
            temperature=float(params.get("temperature", 0.0)),
            remasking=str(params.get("remasking", "low_confidence")),
            tokenizer=self.tokenizer,
            shift=False,
            threshold=None if threshold is None else float(threshold),
            eos_id=self.tokenizer.eos_token_id,
        )
        # generate() returns a plain tensor (not a GenerateOutput).
        ids = out.sequences if hasattr(out, "sequences") else out
        return self._finish(ids, input_ids.shape[1], started)
