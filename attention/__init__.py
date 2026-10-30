"""Standalone Qwen and DeepSeek attention operators."""
from .functional import (
    gqa_decode, gqa_decode_split, gqa_decode_auto, gqa_prefill, kv_append,
    latent_kv_write, mla_query_absorb, mla_value_project, mla_latent_attention,
    mla_decode, mla_prefill, gated_attention, causal_conv1d, causal_conv1d_update,
    delta_recurrent, delta_chunkwise, delta_output, lightning_indexer, topk_select,
    sparse_mla,
)
from .core import is_cuda_extension_loaded
from .reference.gqa import reference_gqa_decode

__all__ = [
    "gqa_decode", "gqa_decode_split", "gqa_decode_auto", "gqa_prefill", "kv_append",
    "is_cuda_extension_loaded", "reference_gqa_decode", "latent_kv_write",
    "mla_query_absorb", "mla_value_project", "mla_latent_attention", "mla_decode",
    "mla_prefill", "gated_attention", "causal_conv1d", "causal_conv1d_update",
    "delta_recurrent", "delta_chunkwise", "delta_output", "lightning_indexer",
    "topk_select", "sparse_mla",
]
__version__ = "0.1.0"
