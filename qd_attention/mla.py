"""MLA forward primitives for normalized latent KV and already-rotated Q/K.

Paged cache uses [Nblocks, page_size, Dc+Dr]; all inputs are contiguous CUDA.
Page IDs, positions, sequence lengths, unique cache slots and selected indices
are caller-owned GPU metadata and must be valid. No backward is supported.
"""
import math
import torch
from .dense import is_cuda_extension_loaded


def _ready():
    if not is_cuda_extension_loaded(): raise RuntimeError('CUDA extension unavailable')


def mla_query_absorb(q_nope, w_uk):
    _ready()
    return torch.ops.qd_attention.mla_project(q_nope, w_uk, True)


def mla_value_project(latent_output, w_uv):
    _ready()
    return torch.ops.qd_attention.mla_project(latent_output, w_uv, False)


def latent_kv_write(latent, rope_key, cache, slot_mapping):
    """In-place write [N,Dc] and [N,Dr] to unique physical int64 slots."""
    _ready()
    torch.ops.qd_attention.latent_kv_write(latent, rope_key, cache, slot_mapping)


def mla_latent_attention(q_absorbed, q_rope, cache, block_table, lengths, positions, *, scale, causal=True, indices=None):
    """Return latent output [B,T,H,Dc]; indices are logical int32 [B,T,K].

    positions are absolute int32 [B,T]. Empty / fully masked rows return zeros.
    -1 sparse indices are ignored; all remaining indices must be unique and valid.
    """
    _ready()
    return torch.ops.qd_attention.mla_attend(q_absorbed, q_rope, cache, block_table, lengths, positions, indices, float(scale), causal)


def mla_decode(q_nope, q_rope, cache, block_table, lengths, positions, w_uk, w_uv, *, scale=None, causal=True, indices=None):
    """Absorbed paged MLA for decode or multiple queries; pre-applied RoPE."""
    if scale is None: scale = 1.0 / math.sqrt(q_nope.shape[-1] + q_rope.shape[-1])
    if (q_nope.ndim == 4 and q_nope.shape[1] == 1 and cache.ndim == 3 and
            block_table.ndim == 2 and w_uk.ndim == 3 and w_uk.shape[-1] <= 512 and
            (indices is not None and indices.ndim == 3 and indices.shape[-1] < 512 or
             indices is None and cache.shape[1] * block_table.shape[1] < 512)):
        _ready()
        return torch.ops.qd_attention.mla_decode_fused(q_nope, q_rope, cache, block_table,
            lengths, positions, w_uk, w_uv, indices, float(scale), causal)
    absorbed = mla_query_absorb(q_nope, w_uk)
    latent = mla_latent_attention(absorbed, q_rope, cache, block_table, lengths, positions, scale=scale, causal=causal, indices=indices)
    return mla_value_project(latent, w_uv)


def mla_prefill(q_nope, q_rope, latent, rope_key, w_uk, w_uv, lengths, positions, *, scale=None):
    """Direct dense MLA with absolute-position causal mask, FP32 accumulators."""
    _ready()
    if scale is None: scale=1.0/math.sqrt(q_nope.shape[-1]+q_rope.shape[-1])
    return torch.ops.qd_attention.mla_prefill_direct(q_nope,q_rope,latent,rope_key,w_uk,w_uv,lengths,positions,float(scale))
