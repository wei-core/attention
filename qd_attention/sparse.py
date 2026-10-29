"""DSA token scoring and selection; sparse attention reuses paged MLA."""
import torch
from .dense import is_cuda_extension_loaded
from .mla import mla_decode


def lightning_indexer(q,k,weights,lengths,positions):
    """Preprocessed q [B,T,H,D], k [B,L,D], weights [B,T,H].

    Caller applies model-specific projections, RoPE, normalization and scaling.
    Lengths and absolute positions are CUDA int32. Masked scores are -inf.
    """
    if not is_cuda_extension_loaded(): raise RuntimeError('CUDA extension unavailable')
    return torch.ops.qd_attention.lightning_indexer(q,k,weights,lengths,positions)


def topk_select(scores,k):
    """Select finite scores descending; ties use smaller logical index, pad -1."""
    if not is_cuda_extension_loaded(): raise RuntimeError('CUDA extension unavailable')
    return torch.ops.qd_attention.topk_select(scores,k)


def sparse_mla(q_nope,q_rope,cache,block_table,lengths,positions,w_uk,w_uv,indices,*,scale=None):
    return mla_decode(q_nope,q_rope,cache,block_table,lengths,positions,w_uk,w_uv,scale=scale,indices=indices)
