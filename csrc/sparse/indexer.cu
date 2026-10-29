#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <limits>
#include <math_constants.h>
namespace {
void check(const torch::Tensor& x,const torch::Tensor& ref) { TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==ref.device(),"all inputs must be contiguous on one CUDA device"); }
template<typename T>
__global__ void index_kernel(const T* q,const T* k,const T* w,const int32_t* lens,const int32_t* positions,float* out,int bsz,int tq,int l,int h,int d) {
 int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x; int64_t n=static_cast<int64_t>(bsz)*tq*l; if(i>=n)return;
 int s=i%l,t=(i/l)%tq,b=i/(static_cast<int64_t>(l)*tq);
 if(s>=lens[b] || s>positions[b*tq+t]) {out[i]=-CUDART_INF_F;return;}
 float score=0; for(int head=0;head<h;head++) {float dot=0; for(int j=0;j<d;j++) dot+=float(q[((static_cast<int64_t>(b)*tq+t)*h+head)*d+j])*float(k[(static_cast<int64_t>(b)*l+s)*d+j]);score+=float(w[(static_cast<int64_t>(b)*tq+t)*h+head])*fmaxf(dot,0.f);}
 out[i]=score;
}
__global__ void topk_kernel(const float* scores,int32_t* out,uint8_t* used,int rows,int l,int k) {
 int row=blockIdx.x*blockDim.x+threadIdx.x; if(row>=rows)return;
 for(int i=0;i<l;i++) used[static_cast<int64_t>(row)*l+i]=0;
 for(int j=0;j<k;j++) {
  float best=-CUDART_INF_F; int bestidx=-1;
  for(int i=0;i<l;i++) {float x=scores[static_cast<int64_t>(row)*l+i]; if(!used[static_cast<int64_t>(row)*l+i] && isfinite(x) && (bestidx<0 || x>best)) {best=x;bestidx=i;} }
  out[static_cast<int64_t>(row)*k+j]=bestidx; if(bestidx>=0) used[static_cast<int64_t>(row)*l+bestidx]=1;
 }
}
}
torch::Tensor lightning_indexer_cuda(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& w,const torch::Tensor& lens,const torch::Tensor& positions) {
 for(const auto& x:{q,k,w,lens,positions})check(x,q);
 TORCH_CHECK(q.dim()==4 && k.dim()==3 && w.dim()==3 && lens.dim()==1 && positions.dim()==2,"q [B,T,H,D], k [B,L,D], weights [B,T,H]");
 TORCH_CHECK(q.size(0)==k.size(0) && q.size(3)==k.size(2) && w.size(0)==q.size(0) && w.size(1)==q.size(1) && w.size(2)==q.size(2) && lens.size(0)==q.size(0) && positions.size(0)==q.size(0) && positions.size(1)==q.size(1),"shape mismatch");
 TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==w.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"dtype mismatch or unsupported");TORCH_CHECK(lens.scalar_type()==at::kInt && positions.scalar_type()==at::kInt,"metadata must be int32");
 auto out=at::empty({q.size(0),q.size(1),k.size(1)},q.options().dtype(at::kFloat));if(!out.numel())return out;
 c10::cuda::CUDAGuard guard(q.device());dim3 grid((out.numel()+127)/128);
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"lightning_indexer",[&]{index_kernel<scalar_t><<<grid,128,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<float>(),q.size(0),q.size(1),k.size(1),q.size(2),q.size(3));});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor topk_select_cuda(const torch::Tensor& scores,int64_t k) {
 TORCH_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim()==3 && scores.scalar_type()==at::kFloat,"scores must be contiguous CUDA float32 [B,T,L]");TORCH_CHECK(k>=0 && k<=std::numeric_limits<int>::max(),"invalid k");
 auto out=at::empty({scores.size(0),scores.size(1),k},scores.options().dtype(at::kInt));if(!out.numel())return out;
 c10::cuda::CUDAGuard guard(scores.device());int rows=scores.size(0)*scores.size(1);auto used=at::empty(scores.sizes(),scores.options().dtype(at::kByte));topk_kernel<<<(rows+127)/128,128,0,at::cuda::getCurrentCUDAStream().stream()>>>(scores.data_ptr<float>(),out.data_ptr<int32_t>(),used.data_ptr<uint8_t>(),rows,scores.size(2),k);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
