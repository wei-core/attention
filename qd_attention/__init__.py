"""Standalone Qwen/DeepSeek Attention CUDA kernels."""

from .dense import gqa_decode, is_cuda_extension_loaded
from .reference.gqa import reference_gqa_decode

__all__ = ["gqa_decode", "reference_gqa_decode", "is_cuda_extension_loaded"]
__version__ = "0.1.0"
