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


def gqa_decode_split(q, k_cache, v_cache, block_table, context_lens, *, splits=4, scale=None, workspace=None, backend="cuda"):
    """Decode independent KV partitions and merge FP32 online-softmax states."""
    if backend == "triton":
        from .triton_decode import paged_decode
        return paged_decode(q, k_cache, v_cache, block_table, context_lens, splits=splits, scale=scale, workspace=workspace)
    if backend != "cuda":
        raise ValueError("backend must be cuda or triton")
    if _LOAD_ERROR is not None:
        raise RuntimeError("qd_attention CUDA extension was not loaded") from _LOAD_ERROR
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if workspace is not None:
        return torch.ops.qd_attention.gqa_decode_split_workspace(q, k_cache, v_cache, block_table, context_lens, float(scale), splits, workspace)
    return torch.ops.qd_attention.gqa_decode_split(q, k_cache, v_cache, block_table, context_lens, float(scale), splits)


def kv_append(key, value, k_cache, v_cache, block_table, positions):
    """Write one token per sequence; caller owns valid, distinct destinations."""
    if _LOAD_ERROR is not None:
        raise RuntimeError("qd_attention CUDA extension was not loaded") from _LOAD_ERROR
    torch.ops.qd_attention.kv_append(key, value, k_cache, v_cache, block_table, positions)


def gqa_prefill(q, k, v, q_lens, kv_lens, *, causal=True, scale=None, backend="cuda"):
    """Dense [B,T,H,D] GQA; causal queries align to the right of valid KV."""
    if backend == "tensorcore":
        from .triton_prefill import tensorcore_prefill
        return tensorcore_prefill(q, k, v, q_lens, kv_lens, causal=causal, scale=scale)
    if backend != "cuda":
        raise ValueError("backend must be cuda or tensorcore")
    if _LOAD_ERROR is not None:
        raise RuntimeError("qd_attention CUDA extension was not loaded") from _LOAD_ERROR
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.qd_attention.gqa_prefill(q, k, v, q_lens, kv_lens, float(scale), causal)


def gqa_decode_auto(q, k_cache, v_cache, block_table, context_lens, *, scale=None):
    """Dispatch from static KV capacity, without synchronizing GPU lengths.

    The conservative split heuristic is an initial policy; callers with measured
    workload-specific choices can use gqa_decode_split directly.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or block_table.ndim != 2:
        raise ValueError('invalid decode input dimensions')
    capacity = block_table.shape[1] * k_cache.shape[1]
    if capacity < 512 or q.shape[0] == 0:
        return gqa_decode(q, k_cache, v_cache, block_table, context_lens, scale=scale)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    desired = max(1, math.ceil(sm_count / (q.shape[0] * max(q.shape[1], 1))))
    split_limit = max(1, capacity // 256)
    splits = min(16, split_limit, 1 << (desired - 1).bit_length())
    return gqa_decode_split(q, k_cache, v_cache, block_table, context_lens,
                            splits=splits, scale=scale, backend='triton')
