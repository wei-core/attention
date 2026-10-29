#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
template <typename T>
__global__ void append_kernel(const T* key, const T* value, T* kc, T* vc,
                              const int32_t* table, const int32_t* pos,
                              int heads, int d, int page_size, int pages) {
    int b = blockIdx.x;
    int t = pos[b];
    int physical = table[static_cast<int64_t>(b) * pages + t / page_size];
    int64_t dst = (static_cast<int64_t>(physical) * page_size + t % page_size) * heads * d;
    int64_t src = static_cast<int64_t>(b) * heads * d;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
        kc[dst+i] = key[src+i];
        vc[dst+i] = value[src+i];
    }
}
__global__ void append_vec16(const uint4* key,const uint4* value,uint4* kc,uint4* vc,const int32_t* table,const int32_t* pos,int vectors,int page_size,int pages){
 int b=blockIdx.x,t=pos[b],physical=table[static_cast<int64_t>(b)*pages+t/page_size];
 int64_t dst=(static_cast<int64_t>(physical)*page_size+t%page_size)*vectors,src=static_cast<int64_t>(b)*vectors;
 for(int i=threadIdx.x;i<vectors;i+=blockDim.x){kc[dst+i]=key[src+i];vc[dst+i]=value[src+i];}
}

}

void kv_append_cuda(const torch::Tensor& key, const torch::Tensor& value,
                    torch::Tensor kc, torch::Tensor vc, const torch::Tensor& table,
                    const torch::Tensor& positions) {
    for (const auto& t : {key, value, kc, vc, table, positions}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "all tensors must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == key.device(), "all tensors must share a device");
    }
    TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(), "key/value must be [B, Hkv, D]");
    TORCH_CHECK(kc.dim() == 4 && vc.sizes() == kc.sizes(), "cache must be [Nblocks, P, Hkv, D]");
    TORCH_CHECK(key.size(1) == kc.size(2) && key.size(2) == kc.size(3), "cache head shape mismatch");
    TORCH_CHECK(kc.size(0) > 0 && kc.size(1) > 0 && key.size(1) > 0 && key.size(2) > 0, "invalid dimensions");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == key.size(0) && table.size(1) > 0 && positions.dim() == 1 && positions.size(0) == key.size(0), "invalid metadata shape");
    TORCH_CHECK(table.scalar_type() == at::kInt && positions.scalar_type() == at::kInt, "metadata must be int32");
    TORCH_CHECK(key.scalar_type() == value.scalar_type() && key.scalar_type() == kc.scalar_type() && key.scalar_type() == vc.scalar_type(), "dtype mismatch");
    TORCH_CHECK(key.scalar_type() == at::kFloat || key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16, "unsupported dtype");
    c10::cuda::CUDAGuard guard(key.device());
    if (key.size(0) == 0) return;
    auto aligned=[](const torch::Tensor& x){return (reinterpret_cast<uintptr_t>(x.data_ptr())&15)==0;};
    int64_t bytes=key.size(1)*key.size(2)*key.element_size();
    if(bytes%16==0 && aligned(key) && aligned(value) && aligned(kc) && aligned(vc)){
        append_vec16<<<key.size(0),256,0,at::cuda::getCurrentCUDAStream().stream()>>>(reinterpret_cast<const uint4*>(key.data_ptr()),reinterpret_cast<const uint4*>(value.data_ptr()),reinterpret_cast<uint4*>(kc.data_ptr()),reinterpret_cast<uint4*>(vc.data_ptr()),table.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),bytes/16,kc.size(1),table.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, key.scalar_type(), "kv_append", [&] {
        append_kernel<scalar_t><<<key.size(0), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(), table.data_ptr<int32_t>(), positions.data_ptr<int32_t>(), key.size(1), key.size(2), kc.size(1), table.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
