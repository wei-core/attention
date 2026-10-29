"""Standalone Qwen/DeepSeek Attention CUDA kernels."""

from .dense import gqa_decode, gqa_decode_split, gqa_decode_auto, kv_append, gqa_prefill, is_cuda_extension_loaded
from .reference.gqa import reference_gqa_decode

__all__ = ["gqa_decode", "gqa_decode_split", "gqa_decode_auto", "kv_append", "gqa_prefill", "reference_gqa_decode", "is_cuda_extension_loaded"]
__version__ = "0.1.0"
