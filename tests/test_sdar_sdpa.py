"""SDPA stand-in mask tests (Experiment 1, OLLADA_SDAR_SDPA=1).

Regression for the 2026-10-25 Dell failure: the first stand-in passed
BlockMask.to_dense() straight to SDPA, but to_dense() returns an int32
BLOCK mask (one entry per block pair), not a per-position mask — SDPA
rejected the dtype, and even a cast would have computed wrong
attention. The fix reconstructs the position mask from the BlockMask's
own mask_mod over an index grid. These tests prove that reconstruction
against the adapter's own _to_block_mask, and that the resulting SDPA
call matches eager flex_attention.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from ollada.adapters.sdar import _sdpa_attn_mask, _to_block_mask


def _loop_mask():
    # Diffusion-like: causal, plus one fully masked row kept finite.
    m = torch.zeros(1, 8, 8)
    for i in range(8):
        for j in range(8):
            m[0, i, j] = 1.0 if j <= i else 0.0
    m[0, 5, :] = 0.0
    m[0, 5, 5] = 1.0
    return m


def test_block_mask_grid_reproduces_loop_mask():
    m = _loop_mask()
    bm = _to_block_mask(m)
    query = torch.zeros(1, 2, 8, 16)
    additive = _sdpa_attn_mask(bm, query)
    assert additive.shape == (1, 1, 8, 8)
    assert additive.dtype == query.dtype
    # 0 = attend, -inf = masked, exactly where the loop mask says so.
    assert torch.equal(additive[0, 0], torch.where(m[0] > 0, 0.0, float("-inf")))


def test_sdpa_with_reconstructed_mask_matches_flex():
    m = _loop_mask()
    bm = _to_block_mask(m)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_attn_mask(bm, q))
    out_flex = flex_attention(q, k, v, block_mask=bm)
    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)


def test_tensor_mask_normalizes_and_rejects_garbage():
    query = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    int_mask = torch.tensor([[[[1, 0], [1, 1]]]])
    additive = _sdpa_attn_mask(int_mask, query)
    assert additive.dtype == torch.bfloat16
    assert additive[0, 0, 0, 1] == float("-inf")
    with pytest.raises(Exception, match="SDPA experiment"):
        _sdpa_attn_mask(object(), query)
