#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/types.h>

#include <cmath>
#include <limits>

#include "../common/block_reduce.cuh"
#include "../common/online_softmax.cuh"
#include <c10/cuda/CUDAException.h>

namespace {

template <typename scalar_t>
__global__ void gqa_paged_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    scalar_t* __restrict__ out, float* __restrict__ partial, int splits,
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
    qd_attention::SoftmaxState state;
    float acc = 0.f;

    const int start = static_cast<int>((static_cast<int64_t>(length) * blockIdx.z) / splits);
    const int end = static_cast<int>((static_cast<int64_t>(length) * (blockIdx.z + 1)) / splits);
    for (int t = start; t < end; ++t) {
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
        float alpha, p;
        state.add(score, alpha, p);
        for (int col = tid; col < d; col += blockDim.x) {
            // This baseline maps one thread to each output feature.
            // The loop executes once for 32 <= D <= 256 and matching CTA.
            acc = acc * alpha + p * static_cast<float>(v[cache_base + col]);
        }
    }
    if (splits == 1) {
        if (tid < d) out[q_base + tid] = static_cast<scalar_t>(state.sum > 0.f ? acc / state.sum : 0.f);
    } else {
        const int64_t base = ((static_cast<int64_t>(b) * hq + h) * splits + blockIdx.z) * (d + 2);
        if (tid == 0) { partial[base] = state.max; partial[base + 1] = state.sum; }
        if (tid < d) partial[base + 2 + tid] = acc;
    }
}


template <typename scalar_t>
__global__ void merge_splits(const float* partial, scalar_t* out, int hq, int d, int splits) {
    int h = blockIdx.x, b = blockIdx.y, col = threadIdx.x;
    qd_attention::SoftmaxState state;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        int64_t base = ((static_cast<int64_t>(b) * hq + h) * splits + s) * (d + 2);
        float other_l = partial[base + 1];
        float a, beta;
        state.merge(partial[base], other_l, a, beta);
        if (col < d) acc = a * acc + beta * partial[base + 2 + col];
    }
    if (col < d) out[(static_cast<int64_t>(b) * hq + h) * d + col] = static_cast<scalar_t>(state.sum > 0.f ? acc / state.sum : 0.f);
}

void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

} // namespace

torch::Tensor gqa_decode_impl(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& context_lens,
    double scale, int64_t splits, const c10::optional<torch::Tensor>& workspace = c10::nullopt) {
    TORCH_CHECK(splits >= 1 && splits <= 65535, "splits must be in [1, 65535]");
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
    const dim3 grid(static_cast<unsigned int>(q.size(1)), static_cast<unsigned int>(q.size(0)), static_cast<unsigned int>(splits));
    torch::Tensor partial;
    if (workspace.has_value()) {
        partial = *workspace;
        TORCH_CHECK(splits > 1, "workspace requires splits > 1");
        TORCH_CHECK(partial.is_cuda() && partial.device() == q.device() && partial.is_contiguous() && partial.scalar_type() == at::kFloat, "workspace must be contiguous float32 on q device");
        TORCH_CHECK(partial.sizes() == at::IntArrayRef({q.size(0), q.size(1), splits, d + 2}), "workspace shape must be [B,Hq,splits,D+2]");
    } else {
        partial = splits > 1 ? at::empty({q.size(0), q.size(1), splits, d + 2}, q.options().dtype(at::kFloat)) : at::empty({0}, q.options().dtype(at::kFloat));
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(),
        "qd_gqa_decode", [&] {
            gqa_paged_decode_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k_cache.data_ptr<scalar_t>(),
                v_cache.data_ptr<scalar_t>(),
                block_table.data_ptr<int32_t>(),
                context_lens.data_ptr<int32_t>(),
                out.data_ptr<scalar_t>(), partial.data_ptr<float>(), static_cast<int>(splits),
                static_cast<int>(q.size(1)),
                static_cast<int>(k_cache.size(2)),
                d,
                static_cast<int>(k_cache.size(1)),
                static_cast<int>(block_table.size(1)),
                static_cast<float>(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if (splits > 1) {
                const dim3 merge_grid(q.size(1), q.size(0));
                merge_splits<scalar_t><<<merge_grid, threads, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<scalar_t>(), q.size(1), d, splits);
            }
        });
    // Surface invalid kernel-launch errors, without synchronizing the GPU.
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "gqa_decode kernel launch failed: ", cudaGetErrorString(status));
    return out;
}


torch::Tensor gqa_decode_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& table, const torch::Tensor& lengths, double scale) {
    return gqa_decode_impl(q, k, v, table, lengths, scale, 1);
}
torch::Tensor gqa_decode_split_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& table, const torch::Tensor& lengths, double scale, int64_t splits) {
    return gqa_decode_impl(q, k, v, table, lengths, scale, splits);
}

torch::Tensor gqa_decode_split_workspace_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& table, const torch::Tensor& lengths, double scale, int64_t splits, const torch::Tensor& workspace) {
    return gqa_decode_impl(q, k, v, table, lengths, scale, splits, workspace);
}
