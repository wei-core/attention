#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include "../common/block_reduce.cuh"
#include "../common/online_softmax.cuh"
namespace {
template<typename T> __global__ void direct(const T* q,const T* qr,const T* latent,const T* rope,const T* wk,const T* wv,const int32_t* lengths,const int32_t* positions,T* out,int tq,int L,int H,int dn,int dr,int dc,int dv,float scale) {
 int h=blockIdx.x,b=blockIdx.y,t=blockIdx.z,tid=threadIdx.x;int end=min(lengths[b],max(0,positions[b*tq+t]+1));
 int64_t row=(static_cast<int64_t>(b)*tq+t)*H+h,qb=row*dn,rb=row*dr;float acc[2]={0.f,0.f};__shared__ float scratch[8];qd_attention::SoftmaxState state;
 for(int s=0;s<end;s++) {
  int64_t cb=(static_cast<int64_t>(b)*L+s)*dc,kb=(static_cast<int64_t>(b)*L+s)*dr;float score=0;
  for(int d=tid;d<dn;d+=blockDim.x){float kval=0;for(int c=0;c<dc;c++)kval+=float(wk[(static_cast<int64_t>(h)*dn+d)*dc+c])*float(latent[cb+c]);score+=float(q[qb+d])*kval;}
  for(int d=tid;d<dr;d+=blockDim.x)score+=float(qr[rb+d])*float(rope[kb+d]);
  score=qd_attention::block_sum(score,scratch)*scale;float a,p;state.add(score,a,p);
  for(int v=tid;v<dv;v+=blockDim.x){float value=0;for(int c=0;c<dc;c++)value+=float(wv[(static_cast<int64_t>(h)*dv+v)*dc+c])*float(latent[cb+c]);acc[v/blockDim.x]=a*acc[v/blockDim.x]+p*value;}
 }
 for(int v=tid;v<dv;v+=blockDim.x)out[row*dv+v]=static_cast<T>(state.sum>0?acc[v/blockDim.x]/state.sum:0.f);
}
}
torch::Tensor mla_prefill_direct_cuda(const torch::Tensor& q,const torch::Tensor& qr,const torch::Tensor& latent,const torch::Tensor& rope,const torch::Tensor& wk,const torch::Tensor& wv,const torch::Tensor& lens,const torch::Tensor& positions,double scale) {
 for(const auto& x:{q,qr,latent,rope,wk,wv,lens,positions})TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device(),"all inputs must be contiguous on one CUDA device");
 TORCH_CHECK(q.dim()==4 && qr.dim()==4 && latent.dim()==3 && rope.dim()==3 && wk.dim()==3 && wv.dim()==3 && lens.dim()==1 && positions.dim()==2,"invalid tensor ranks");
 TORCH_CHECK(qr.size(0)==q.size(0) && qr.size(1)==q.size(1) && qr.size(2)==q.size(2) && latent.size(0)==q.size(0) && rope.size(0)==q.size(0) && rope.size(1)==latent.size(1) && rope.size(2)==qr.size(3),"invalid input shapes");
 TORCH_CHECK(wk.size(0)==q.size(2) && wk.size(1)==q.size(3) && wk.size(2)==latent.size(2) && wv.size(0)==q.size(2) && wv.size(2)==latent.size(2),"invalid projection shapes");
 TORCH_CHECK(q.size(2)>0 && q.size(3)>0 && qr.size(3)>0 && latent.size(2)>0 && wv.size(1)>0 && wv.size(1)<=512 && q.size(0)<=65535 && q.size(1)<=65535,"unsupported dimensions");
 TORCH_CHECK(lens.size(0)==q.size(0) && positions.size(0)==q.size(0) && positions.size(1)==q.size(1) && lens.scalar_type()==at::kInt && positions.scalar_type()==at::kInt,"invalid metadata");
 TORCH_CHECK(q.scalar_type()==qr.scalar_type() && q.scalar_type()==latent.scalar_type() && q.scalar_type()==rope.scalar_type() && q.scalar_type()==wk.scalar_type() && q.scalar_type()==wv.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"dtype mismatch or unsupported");TORCH_CHECK(std::isfinite(scale) && scale>0,"invalid scale");
 auto out=at::empty({q.size(0),q.size(1),q.size(2),wv.size(1)},q.options());if(!out.numel())return out;c10::cuda::CUDAGuard guard(q.device());dim3 grid(q.size(2),q.size(0),q.size(1));
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"mla_prefill_direct",[&]{direct<scalar_t><<<grid,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),qr.data_ptr<scalar_t>(),latent.data_ptr<scalar_t>(),rope.data_ptr<scalar_t>(),wk.data_ptr<scalar_t>(),wv.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<scalar_t>(),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
