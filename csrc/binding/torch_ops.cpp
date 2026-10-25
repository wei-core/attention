#include <torch/extension.h>
#include <torch/library.h>

// Implementation defined in csrc/dense/gqa_decode.cu.
torch::Tensor gqa_decode_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& context_lens,
    double scale);

TORCH_LIBRARY(qd_attention, m) {
    m.def("gqa_decode(Tensor q, Tensor k_cache, Tensor v_cache, "
          "Tensor block_table, Tensor context_lens, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(qd_attention, CUDA, m) {
    m.impl("gqa_decode", TORCH_FN(gqa_decode_cuda));
}

// Imports this extension as a Python module; ops are exposed via torch.ops.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
