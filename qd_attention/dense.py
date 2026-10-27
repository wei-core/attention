"""Public dense Attention API (forward / inference only in v0.1)."""

import math
import torch

try:
    from . import _C as _cuda_extension  # noqa: F401 ; registers torch.ops.qd_attention
    _LOAD_ERROR = None
except ImportError as exc:
    _LOAD_ERROR = exc


def is_cuda_extension_loaded() -> bool:
    return _LOAD_ERROR is None


def gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Paged single-token MHA/MQA/GQA decode; *unoptimized* CUDA baseline.

    Inputs are CUDA and contiguous; Q/K/V share dtype. No autograd support.
    See docs/KERNEL_CONTRACT.md for full layout and constraints.
    """
    if _LOAD_ERROR is not None:
        raise RuntimeError(
            "qd_attention CUDA extension was not loaded. "
            "Install using `pip install -v --no-build-isolation -e .` "
            "on a machine with nvcc and a CUDA-enabled PyTorch build."
        ) from _LOAD_ERROR
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.qd_attention.gqa_decode(
        q, k_cache, v_cache, block_table, context_lens, float(scale)
    )


def gqa_decode_split(q, k_cache, v_cache, block_table, context_lens, *, splits=4, scale=None):
    """Decode independent KV partitions and merge FP32 online-softmax states."""
    if _LOAD_ERROR is not None:
        raise RuntimeError("qd_attention CUDA extension was not loaded") from _LOAD_ERROR
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.qd_attention.gqa_decode_split(q, k_cache, v_cache, block_table, context_lens, float(scale), splits)


def kv_append(key, value, k_cache, v_cache, block_table, positions):
    """Write one token per sequence; caller owns valid, distinct destinations."""
    if _LOAD_ERROR is not None:
        raise RuntimeError("qd_attention CUDA extension was not loaded") from _LOAD_ERROR
    torch.ops.qd_attention.kv_append(key, value, k_cache, v_cache, block_table, positions)
