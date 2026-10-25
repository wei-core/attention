"""Slow, explicit paged GQA math oracle. CPU and CUDA compatible."""

import math
import torch


def reference_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference for one-token attention; no CUDA extension needed.

    Reassembles logical K/V from block table. Does attention arithmetic in FP32.
    This is intentionally not suitable as a performance benchmark.
    """
    bsz, hq, d = q.shape
    nblocks, block_size, hkv, cache_d = k_cache.shape
    assert cache_d == d
    assert v_cache.shape == k_cache.shape
    assert hq % hkv == 0
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    outputs = []
    for b in range(bsz):
        length = int(context_lens[b].item())
        if length == 0:
            outputs.append(torch.zeros_like(q[b]))
            continue
        n_pages = (length + block_size - 1) // block_size
        pages = block_table[b, :n_pages].long()
        keys = k_cache.index_select(0, pages).reshape(-1, hkv, d)[:length]
        values = v_cache.index_select(0, pages).reshape(-1, hkv, d)[:length]
        keys = keys.transpose(0, 1).repeat_interleave(hq // hkv, dim=0).float()
        values = values.transpose(0, 1).repeat_interleave(hq // hkv, dim=0).float()
        scores = torch.einsum("hd,hld->hl", q[b].float(), keys) * scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("hl,hld->hd", weights, values)
        outputs.append(output.to(q.dtype))
    return torch.stack(outputs, dim=0)
