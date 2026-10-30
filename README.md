# Attention

CUDA and Triton attention operators for Qwen and DeepSeek workloads. The library
includes paged grouped-query decode, dense prefill, latent attention, gated and
delta primitives, and sparse indexing. These are forward-only research operators;
backend support and performance depend on the operator and input shape.

## Installation

Use Linux or WSL2, an NVIDIA GPU, a CUDA toolkit compatible with the installed
CUDA-enabled PyTorch, and a C++17 compiler. For RTX 5060/5090 use CUDA 12.8 or
newer and set the architecture to 12.0.

```bash
export TORCH_CUDA_ARCH_LIST=12.0
python -m pip install -v --no-build-isolation -e .
```

Install Triton separately for the optional Triton backends. The build requires
PyTorch to be installed in the active environment before installation.

## Usage

```python
import attention

# q: [batch, query_heads, head_dim]
# k_cache/v_cache: [blocks, block_size, kv_heads, head_dim]
out = attention.gqa_decode(q, k_cache, v_cache, block_table, context_lens)
```

The existing `qd_attention` import remains available for compatibility. Input
metadata and cache tensors must satisfy the requirements checked by each CUDA
operator. The operators do not implement backward propagation. Benchmark scripts
are in `benchmarks/`; tests and detailed engineering records are maintained
separately.
