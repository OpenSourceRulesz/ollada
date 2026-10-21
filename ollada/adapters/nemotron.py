"""Nemotron-Labs Diffusion family adapter (NVIDIA).

One checkpoint, three modes — the recipe picks the default:
  diffusion   -> model.generate(ids, block_length, threshold, eos_token_id)
  ar          -> model.ar_generate(ids)
  linear_spec -> model.linear_spec_generate(ids, block_length, eos_token_id)
Each returns (out_ids, nfe). Requires transformers>=5.0.0 (recipe pin) and
therefore runs in the isolated nemotron worker env (PLAN.md section 3).

UNVERIFIED in ollada until a real-load gate passes on a T4.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModel, AutoTokenizer

from ollada.adapters.base import BaseAdapter, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class NemotronAdapter(BaseAdapter):
    family = "nemotron"

    def load(self) -> None:
        recipe = self.recipe
        self.tokenizer = AutoTokenizer.from_pretrained(recipe.source, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        qc = quantization_config(recipe.quant_default)
        if qc is not None:
            kwargs["quantization_config"] = qc
            kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(recipe.source, **kwargs)
        if qc is None:
            dtype = _DTYPES.get(recipe.dtype, torch.bfloat16)
            model = model.to(dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        self.model = model.eval()

    @torch.no_grad()
    def chat(self, messages, overrides=None):
        self._require_loaded()
        params = self._params(overrides)
        input_ids = self._prompt_ids(messages).to(self.model.device)
        mode = str(params.get("mode", "diffusion"))
        max_new = int(params.get("max_new_tokens", 512))
        eos = self.tokenizer.eos_token_id
        started = time.time()
        if mode == "ar":
            out_ids, nfe = self.model.ar_generate(input_ids, max_new_tokens=max_new)
        elif mode == "linear_spec":
            out_ids, nfe = self.model.linear_spec_generate(
                input_ids,
                max_new_tokens=max_new,
                block_length=int(params.get("block_length", 32)),
                eos_token_id=eos,
            )
        else:
            out_ids, nfe = self.model.generate(
                input_ids,
                max_new_tokens=max_new,
                block_length=int(params.get("block_length", 32)),
                threshold=float(params.get("threshold", 0.9)),
                eos_token_id=eos,
            )
        return self._finish(out_ids, input_ids.shape[1], started, nfe=int(nfe))
