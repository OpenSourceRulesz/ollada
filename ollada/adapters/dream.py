"""Dream family adapter (Dream-v0-Instruct-7B).

Load path mirrors the proven npc-engine harness: BitsAndBytesConfig for
8-bit (never a raw load_in_8bit kwarg) and the transformers==4.53.3 pin
from the recipe (5.x breaks Dream's RoPE config with KeyError 'default').

UNVERIFIED in ollada until the Phase 1 real-load gate passes on a T4.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModel, AutoTokenizer

from ollada.adapters.base import BaseAdapter, GenResult, quantization_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class DreamAdapter(BaseAdapter):
    family = "dream"

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
        self.model = AutoModel.from_pretrained(recipe.source, **kwargs)
        if qc is None and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages, overrides=None):
        self._require_loaded()
        params = self._params(overrides)
        input_ids = self._prompt_ids(messages).to(self.model.device)
        started = time.time()
        out = self.model.diffusion_generate(
            input_ids,
            max_new_tokens=int(params.get("gen_length", 256)),
            steps=int(params.get("steps", 32)),
            temperature=float(params.get("temperature", 0.2)),
            top_p=float(params.get("top_p", 0.95)),
            alg="origin",
            alg_temp=0.0,
            mask_id=self.model.config.mask_token_id,
        )
        ids = out.sequences if hasattr(out, "sequences") else out
        return self._finish(ids, input_ids.shape[1], started)
