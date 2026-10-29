"""Standalone Qwen/DeepSeek Attention CUDA kernels."""

from .dense import gqa_decode, gqa_decode_split, gqa_decode_auto, kv_append, gqa_prefill, is_cuda_extension_loaded
from .reference.gqa import reference_gqa_decode

__all__ = ["gqa_decode", "gqa_decode_split", "gqa_decode_auto", "kv_append", "gqa_prefill", "reference_gqa_decode", "is_cuda_extension_loaded"]
__version__ = "0.1.0"

from .mla import (latent_kv_write, mla_query_absorb, mla_value_project, mla_latent_attention, mla_decode, mla_prefill)
from .qwen import (gated_attention, causal_conv1d, causal_conv1d_update, delta_recurrent, delta_chunkwise, delta_output)
from .sparse import lightning_indexer, topk_select, sparse_mla

__all__ += ["latent_kv_write", "mla_query_absorb", "mla_value_project", "mla_latent_attention", "mla_decode", "mla_prefill", "gated_attention", "causal_conv1d", "causal_conv1d_update", "delta_recurrent", "delta_chunkwise", "delta_output", "lightning_indexer", "topk_select", "sparse_mla"]
