#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cmath>
#include <limits>

#include "../common/block_reduce.cuh"

namespace {

template <typename scalar_t>
__global__ void gqa_paged_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    scalar_t* __restrict__ out,
    int hq, int hkv, int d, int block_size, int max_pages,
    float scale) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head = h / (hq / hkv);
    const int length = context_lens[b];
    const int64_t q_base = (static_cast<int64_t>(b) * hq + h) * d;
    const int64_t pages_base = static_cast<int64_t>(b) * max_pages;

    __shared__ float warp_sums[8];
    float m = -CUDART_INF_F;
    float normalizer = 0.f;
    float acc = 0.f;

    for (int t = 0; t < length; ++t) {
        const int page = t / block_size;
        const int slot = t % block_size;
        const int32_t physical_block = block_table[pages_base + page];
        const int64_t cache_base =
            ((static_cast<int64_t>(physical_block) * block_size + slot) * hkv + kv_head) * d;
        float dot = 0.f;
        for (int col = tid; col < d; col += blockDim.x) {
            dot += static_cast<float>(q[q_base + col]) *
                   static_cast<float>(k[cache_base + col]);
        }
        const float score = qd_attention::block_sum(dot, warp_sums) * scale;

        // Streaming log-sum-exp: identical scalar state in every thread.
        const float new_m = fmaxf(m, score);
        const float alpha = expf(m - new_m);
        const float p = expf(score - new_m);
        normalizer = normalizer * alpha + p;
        for (int col = tid; col < d; col += blockDim.x) {
            // This baseline maps one thread to each output feature.
            // The loop executes once for 32 <= D <= 256 and matching CTA.
            acc = acc * alpha + p * static_cast<float>(v[cache_base + col]);
        }
        m = new_m;
    }
    for (int col = tid; col < d; col += blockDim.x) {
        out[q_base + col] = static_cast<scalar_t>(normalizer > 0.f ? acc / normalizer : 0.f);
    }
}

void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

} // namespace

torch::Tensor gqa_decode_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& context_lens,
    double scale) {
    check_cuda_contiguous(q, "q");
    check_cuda_contiguous(k_cache, "k_cache");
    check_cuda_contiguous(v_cache, "v_cache");
    check_cuda_contiguous(block_table, "block_table");
    check_cuda_contiguous(context_lens, "context_lens");
    TORCH_CHECK(q.dim() == 3, "q must be [B, Hq, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [Nblocks, BlockSize, Hkv, D]");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache shape mismatch");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, MaxPages]");
    TORCH_CHECK(context_lens.dim() == 1, "context_lens must be [B]");
    TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device() &&
                q.device() == block_table.device() && q.device() == context_lens.device(),
                "all tensors must use one CUDA device");
    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() &&
                q.scalar_type() == v_cache.scalar_type(), "q/k/v dtype must match");
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf ||
                q.scalar_type() == at::kBFloat16, "q/k/v must be float32/float16/bfloat16");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
                "block_table and context_lens must be int32");
    TORCH_CHECK(q.size(0) == block_table.size(0) && q.size(0) == context_lens.size(0),
                "batch sizes must match");
    TORCH_CHECK(q.size(2) == k_cache.size(3), "q head_dim and cache head_dim must match");
    TORCH_CHECK(q.size(1) > 0 && k_cache.size(2) > 0 &&
                q.size(1) % k_cache.size(2) == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(q.size(2) > 0 && q.size(2) <= 256, "supported head_dim is [1, 256]");
    TORCH_CHECK(k_cache.size(0) > 0 && k_cache.size(1) > 0,
                "cache must have positive blocks and block_size");
    TORCH_CHECK(block_table.size(1) > 0, "block_table must have at least one page");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be positive and finite");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() &&
                q.size(1) <= std::numeric_limits<int>::max() &&
                k_cache.size(2) <= std::numeric_limits<int>::max() &&
                k_cache.size(1) <= std::numeric_limits<int>::max() &&
                block_table.size(1) <= std::numeric_limits<int>::max(),
                "tensor sizes exceed int32 CUDA launch limits");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto out = at::empty_like(q);
    if (q.size(0) == 0) return out;

    const int d = static_cast<int>(q.size(2));
    const int threads = d <= 32 ? 32 : d <= 64 ? 64 : d <= 128 ? 128 : 256;
    const dim3 grid(static_cast<unsigned int>(q.size(1)), static_cast<unsigned int>(q.size(0)));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(),
        "qd_gqa_decode", [&] {
            gqa_paged_decode_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k_cache.data_ptr<scalar_t>(),
                v_cache.data_ptr<scalar_t>(),
                block_table.data_ptr<int32_t>(),
                context_lens.data_ptr<int32_t>(),
                out.data_ptr<scalar_t>(),
                static_cast<int>(q.size(1)),
                static_cast<int>(k_cache.size(2)),
                d,
                static_cast<int>(k_cache.size(1)),
                static_cast<int>(block_table.size(1)),
                static_cast<float>(scale));
        });
    // Surface invalid kernel-launch errors, without synchronizing the GPU.
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "gqa_decode kernel launch failed: ", cudaGetErrorString(status));
    return out;
}
