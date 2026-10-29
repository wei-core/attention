#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include "../common/block_reduce.cuh"
#include "../common/online_softmax.cuh"
#include "../common/tile_utils.cuh"

namespace {
// One CTA per query row/head; K/V are streamed in fixed-size tiles and no
// quadratic score tensor is materialized. Arithmetic and state are FP32.
template <typename T>
__global__ void prefill_kernel(const T* q, const T* k, const T* v,
        const int32_t* qlens, const int32_t* klens, T* out,
        int tq, int tk, int hq, int hk, int d, float scale, bool causal) {
    const int h = blockIdx.x, row = blockIdx.y, b = blockIdx.z, col = threadIdx.x;
    const int qlen = qlens[b], klen = klens[b];
    const int64_t qbase = ((static_cast<int64_t>(b) * tq + row) * hq + h) * d;
    if (row >= qlen) { if (col < d) out[qbase + col] = T(0); return; }
    const int end = causal ? qd_attention::causal_end(row, qlen, klen) : klen;
    const int kh = qd_attention::head_to_kv_head(h, hq, hk);
    const float query = col < d ? static_cast<float>(q[qbase + col]) : 0.f;
    __shared__ float reduction[8];
    qd_attention::SoftmaxState state;
    float acc = 0.f;
    constexpr int tile_n = 32;
    for (int tile = 0; tile < end; tile += tile_n) {
        const int stop = min(tile + tile_n, end);
        for (int t = tile; t < stop; ++t) {
            const int64_t base = ((static_cast<int64_t>(b) * tk + t) * hk + kh) * d;
            float x = col < d ? query * static_cast<float>(k[base + col]) : 0.f;
            float score = qd_attention::block_sum(x, reduction) * scale;
            float a, beta;
            state.add(score, a, beta);
            if (col < d) acc = a * acc + beta * static_cast<float>(v[base + col]);
        }
    }
    if (col < d) out[qbase + col] = static_cast<T>(state.sum > 0.f ? acc / state.sum : 0.f);
}
}

torch::Tensor gqa_prefill_cuda(const torch::Tensor& q, const torch::Tensor& k,
        const torch::Tensor& v, const torch::Tensor& qlens, const torch::Tensor& klens,
        double scale, bool causal) {
    for (const auto& x : {q, k, v, qlens, klens}) {
        TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "all inputs must be contiguous CUDA tensors");
        TORCH_CHECK(x.device() == q.device(), "all inputs must be on the same device");
    }
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "q/k/v must be [B,T,H,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3), "batch or head_dim mismatch");
    TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0 && q.size(2) % k.size(2) == 0, "invalid head mapping");
    TORCH_CHECK(q.size(3) > 0 && q.size(3) <= 256, "head_dim must be in [1,256]");
    TORCH_CHECK(q.size(0) <= 65535 && q.size(1) <= 65535 && q.size(2) <= 2147483647 && k.size(1) <= 2147483647, "dimensions exceed launch limits");
    TORCH_CHECK(qlens.dim() == 1 && klens.dim() == 1 && qlens.size(0) == q.size(0) && klens.size(0) == q.size(0), "lengths must be [B]");
    TORCH_CHECK(qlens.scalar_type() == at::kInt && klens.scalar_type() == at::kInt, "lengths must be int32");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "unsupported dtype");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be positive and finite");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty_like(q);
    if (!q.size(0) || !q.size(1)) return out;
    int d = q.size(3), threads = d <= 32 ? 32 : d <= 64 ? 64 : d <= 128 ? 128 : 256;
    dim3 grid(q.size(2), q.size(1), q.size(0));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "gqa_prefill", [&] {
        prefill_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            qlens.data_ptr<int32_t>(), klens.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
            q.size(1), k.size(1), q.size(2), k.size(2), d, scale, causal);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
