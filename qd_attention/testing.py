"""Deterministic data generation for correctness and microbenchmarks."""

from __future__ import annotations
import math
import torch


def make_decode_case(
    *,
    lengths: list[int],
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 42,
):
    """Returns (q, k_cache, v_cache, block_table, context_lens).

    Pages for distinct batches are deliberately permuted, to expose bugs that
    incorrectly assume logical/physical cache-block order is identical.
    """
    assert lengths and all(x >= 0 for x in lengths)
    assert q_heads % kv_heads == 0
    max_pages = max(1, math.ceil(max(lengths) / block_size))
    num_blocks = len(lengths) * max_pages
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(len(lengths), q_heads, head_dim, generator=gen, dtype=torch.float32)
    k = torch.randn(num_blocks, block_size, kv_heads, head_dim, generator=gen)
    v = torch.randn(num_blocks, block_size, kv_heads, head_dim, generator=gen)
    order = torch.randperm(num_blocks, generator=gen, dtype=torch.int64).view(len(lengths), max_pages)
    table = order.to(torch.int32)
    for b, length in enumerate(lengths):
        used_pages = math.ceil(length / block_size)
        table[b, used_pages:] = -1
    return (
        q.to(device=device, dtype=dtype).contiguous(),
        k.to(device=device, dtype=dtype).contiguous(),
        v.to(device=device, dtype=dtype).contiguous(),
        table.to(device=device).contiguous(),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )
