"""Stable-DiffCoder family adapter (ByteDance-Seed).

Reference call shape (from the model card):
  model.generate(..., steps=512, gen_length=512, block_length=4,
                 temperature=0.0, remasking='low_confidence')
Our recipe defaults are the provisional, to-be-swept values in the catalog.

UNVERIFIED in ollada until a real-load gate passes on a T4.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ollada.adapters.base import BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class DiffCoderAdapter(BaseAdapter):
    family = "diffcoder"

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

    @torch.no_grad()
    def chat(self, messages, overrides=None):
        self._require_loaded()
        params = self._params(overrides)
        input_ids = self._prompt_ids(messages).to(self.model.device)
        started = time.time()
        out = self.model.generate(
            input_ids,
            steps=int(params.get("steps", 128)),
            gen_length=int(params.get("gen_length", 256)),
            block_length=int(params.get("block_length", 4)),
            temperature=float(params.get("temperature", 0.0)),
            remasking=str(params.get("remasking", "low_confidence")),
        )
        ids = out.sequences if hasattr(out, "sequences") else out
        return self._finish(ids, input_ids.shape[1], started)
