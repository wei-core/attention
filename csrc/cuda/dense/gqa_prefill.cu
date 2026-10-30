#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
template<typename T> __device__ T from_float(float x);
template<> __device__ __half from_float<__half>(float x){return __float2half_rn(x);}
template<> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x){return __float2bfloat16_rn(x);}
// 16 query rows per CTA, 16 keys per tile. WMMA forms QK and PV;
// FP32 online softmax and output accumulation live in shared memory.
template<typename T> __global__ void prefill_wmma(const T* q,const T* k,const T* v,const int32_t* qlens,const int32_t* klens,T* out,int tq,int tk,int hq,int hk,int d,float scale,bool causal) {
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 using namespace nvcuda;
 int h=blockIdx.x,b=blockIdx.z,qstart=blockIdx.y*16,lane=threadIdx.x;
 int qlen=qlens[b],klen=klens[b],kh=qd_attention::head_to_kv_head(h,hq,hk);
 __shared__ __align__(32) T a[256],bb[256],prob[256];
 __shared__ __align__(32) float tile[256],acc[16*256];
 __shared__ float mx[16],den[16],alpha[16];
 for(int x=lane;x<16*d;x+=32)acc[x]=0;
 if(lane<16){mx[lane]=-CUDART_INF_F;den[lane]=0;}
 __syncwarp();
 int tile_end=causal?(qstart>=qlen?0:qd_attention::causal_end(min(qstart+15,qlen-1),qlen,klen)):klen;
 for(int start=0;start<tile_end;start+=16){
  wmma::fragment<wmma::accumulator,16,16,16,float> score;
  wmma::fill_fragment(score,0.f);
  for(int ds=0;ds<d;ds+=16){
   for(int x=lane;x<256;x+=32){int r=x/16,c=x%16;int row=qstart+r,dim=ds+c,key=start+r;
    a[x]=(row<qlen && row<tq && dim<d)?q[((static_cast<int64_t>(b)*tq+row)*hq+h)*d+dim]:from_float<T>(0.f);
    bb[x]=(key<klen && dim<d)?k[((static_cast<int64_t>(b)*tk+key)*hk+kh)*d+dim]:from_float<T>(0.f);
   }
   __syncwarp();
   wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> qa;
   wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> kb;
   wmma::load_matrix_sync(qa,a,16);wmma::load_matrix_sync(kb,bb,16);wmma::mma_sync(score,qa,kb,score);__syncwarp();
  }
  wmma::store_matrix_sync(tile,score,16,wmma::mem_row_major);__syncwarp();
  if(lane<16){int row=qstart+lane;int end=causal?qd_attention::causal_end(row,qlen,klen):klen;
   float m=mx[lane];
   for(int j=0;j<16;j++)if(row<qlen && row<tq && start+j<end)m=fmaxf(m,tile[lane*16+j]*scale);
   float old=den[lane]>0?expf(mx[lane]-m):0.f, sum=0;
   for(int j=0;j<16;j++){float p=(row<qlen && row<tq && start+j<end)?expf(tile[lane*16+j]*scale-m):0.f;prob[lane*16+j]=from_float<T>(p);sum+=p;}
   alpha[lane]=old;den[lane]=den[lane]*old+sum;mx[lane]=m;
  }
  __syncwarp();
  for(int x=lane;x<16*d;x+=32)acc[x]*=alpha[x/d];
  __syncwarp();
  for(int ds=0;ds<d;ds+=16){
   for(int x=lane;x<256;x+=32){int key=start+x/16,dim=ds+x%16;bb[x]=(key<klen && dim<d)?v[((static_cast<int64_t>(b)*tk+key)*hk+kh)*d+dim]:from_float<T>(0.f);}
   __syncwarp();
   wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> pa;
   wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> vb;
   wmma::fragment<wmma::accumulator,16,16,16,float> product;
   wmma::fill_fragment(product,0.f);wmma::load_matrix_sync(pa,prob,16);wmma::load_matrix_sync(vb,bb,16);wmma::mma_sync(product,pa,vb,product);wmma::store_matrix_sync(tile,product,16,wmma::mem_row_major);__syncwarp();
   for(int x=lane;x<256;x+=32){int dim=ds+x%16;if(dim<d)acc[(x/16)*d+dim]+=tile[x];}
   __syncwarp();
  }
 }
 for(int x=lane;x<16*d;x+=32){int row=qstart+x/d,dim=x%d;if(row<tq)out[((static_cast<int64_t>(b)*tq+row)*hq+h)*d+dim]=from_float<T>(den[x/d]>0?acc[x]/den[x/d]:0.f);}
 #endif
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
    if ((q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16) && q.size(1)>128 && at::cuda::getCurrentDeviceProperties()->major>=8) {
        dim3 tiled(q.size(2),(q.size(1)+15)/16,q.size(0));
        auto stream=at::cuda::getCurrentCUDAStream().stream();
        if(q.scalar_type()==at::kHalf) prefill_wmma<__half><<<tiled,32,0,stream>>>(reinterpret_cast<const __half*>(q.data_ptr()),reinterpret_cast<const __half*>(k.data_ptr()),reinterpret_cast<const __half*>(v.data_ptr()),qlens.data_ptr<int32_t>(),klens.data_ptr<int32_t>(),reinterpret_cast<__half*>(out.data_ptr()),q.size(1),k.size(1),q.size(2),k.size(2),d,scale,causal);
        else prefill_wmma<__nv_bfloat16><<<tiled,32,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),qlens.data_ptr<int32_t>(),klens.data_ptr<int32_t>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),q.size(1),k.size(1),q.size(2),k.size(2),d,scale,causal);
        C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "gqa_prefill", [&] {
        prefill_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            qlens.data_ptr<int32_t>(), klens.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
            q.size(1), k.size(1), q.size(2), k.size(2), d, scale, causal);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
