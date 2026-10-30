#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include "../common/block_reduce.cuh"
namespace {
void valid(const torch::Tensor& x,const torch::Tensor& y) { TORCH_CHECK(x.is_cuda() && y.is_cuda() && x.is_contiguous() && y.is_contiguous() && x.device()==y.device(),"inputs must be contiguous on one CUDA device"); }
void supported(const torch::Tensor& x) { TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16,"unsupported dtype"); }
template<typename T>
__global__ void gate_kernel(const T* a,const T* g,T* out,int64_t n) {
 int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n) { float gate=float(g[i]); out[i]=static_cast<T>(float(a[i])/(1.f+expf(-gate))); }
}
template<typename T>
__global__ void norm_kernel(const T* x,const T* z,const T* weight,T* out,int d,float eps) {
 int row=blockIdx.x,tid=threadIdx.x; __shared__ float scratch[8]; float sum=0;
 for(int i=tid;i<d;i+=blockDim.x) {float v=float(x[static_cast<int64_t>(row)*d+i]);sum+=v*v;}
 float inv=rsqrtf(qd_attention::block_sum(sum,scratch)/d+eps);
 for(int i=tid;i<d;i+=blockDim.x) {int64_t at=static_cast<int64_t>(row)*d+i;float g=float(z[at]);out[at]=static_cast<T>(float(x[at])*inv*float(weight[i])*g/(1.f+expf(-g)));}
}
template<typename T>
__global__ void delta_kernel(const T* q,const T* k,const T* v,const float* g,const float* beta,const float* old,float* next,T* out,int dk,int dv,float eps) {
 int bh=blockIdx.x, col=threadIdx.x; if(col>=dv) return;
 int64_t qb=static_cast<int64_t>(bh)*dk, vb=static_cast<int64_t>(bh)*dv, sb=static_cast<int64_t>(bh)*dk*dv;
 float q2=0,k2=0; for(int i=0;i<dk;i++) {float a=float(q[qb+i]),b=float(k[qb+i]);q2+=a*a;k2+=b*b;}
 float qscale=rsqrtf(q2+eps)*rsqrtf(float(dk)),kscale=rsqrtf(k2+eps),decay=expf(g[bh]),pred=0;
 for(int i=0;i<dk;i++) pred+=float(k[qb+i])*kscale*decay*old[sb+static_cast<int64_t>(i)*dv+col];
 float delta=beta[bh]*(float(v[vb+col])-pred),y=0;
 for(int i=0;i<dk;i++) {int64_t ix=sb+static_cast<int64_t>(i)*dv+col;float s=decay*old[ix]+float(k[qb+i])*kscale*delta;next[ix]=s;y+=float(q[qb+i])*qscale*s;}
 out[vb+col]=static_cast<T>(y);
}
}
torch::Tensor gated_attention_cuda(const torch::Tensor& a,const torch::Tensor& gate) {
 valid(a,gate); supported(a); TORCH_CHECK(a.sizes()==gate.sizes() && a.scalar_type()==gate.scalar_type(),"shape or dtype mismatch"); auto out=at::empty_like(a); if(!a.numel()) return out;
 c10::cuda::CUDAGuard guard(a.device()); AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,a.scalar_type(),"gated_attention",[&] {gate_kernel<scalar_t><<<(a.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(a.data_ptr<scalar_t>(),gate.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),a.numel());}); C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor delta_output_cuda(const torch::Tensor& x,const torch::Tensor& z,const torch::Tensor& weight,double eps) {
 valid(x,z);valid(x,weight);supported(x);TORCH_CHECK(x.sizes()==z.sizes() && x.dim()>=1 && weight.dim()==1 && weight.size(0)==x.size(-1) && x.size(-1)>0,"invalid shape");TORCH_CHECK(x.scalar_type()==z.scalar_type() && x.scalar_type()==weight.scalar_type(),"dtype mismatch");TORCH_CHECK(std::isfinite(eps) && eps>0,"eps must be positive and finite"); auto out=at::empty_like(x);if(!x.numel())return out;
 c10::cuda::CUDAGuard guard(x.device());int d=x.size(-1);AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"delta_output",[&]{norm_kernel<scalar_t><<<x.numel()/d,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(x.data_ptr<scalar_t>(),z.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),d,eps);});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::tuple<torch::Tensor,torch::Tensor> delta_recurrent_cuda(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& g,const torch::Tensor& beta,const torch::Tensor& state,double eps) {
 for(const auto& x:{k,v,g,beta,state})valid(q,x);supported(q);
 TORCH_CHECK(q.dim()==3 && k.sizes()==q.sizes() && v.dim()==3 && v.size(0)==q.size(0) && v.size(1)==q.size(1),"q/k [B,H,Dk], v [B,H,Dv]");
 TORCH_CHECK(q.size(2)>0 && v.size(2)>0 && v.size(2)<=1024 && state.dim()==4 && state.size(0)==q.size(0) && state.size(1)==q.size(1) && state.size(2)==q.size(2) && state.size(3)==v.size(2),"invalid state dimensions");
 TORCH_CHECK(g.dim()==2 && beta.sizes()==g.sizes() && g.size(0)==q.size(0) && g.size(1)==q.size(1),"g/beta must be [B,H]");
 TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==v.scalar_type() && state.scalar_type()==at::kFloat && g.scalar_type()==at::kFloat && beta.scalar_type()==at::kFloat,"q/k/v dtype must match; state/g/beta must be float32");
 TORCH_CHECK(std::isfinite(eps) && eps>0,"eps must be positive and finite");auto out=at::empty_like(v),next=at::empty_like(state); if(!q.numel())return {out,next};
 c10::cuda::CUDAGuard guard(q.device());int threads=((v.size(2)+31)/32)*32;AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"delta_recurrent",[&]{delta_kernel<scalar_t><<<q.size(0)*q.size(1),threads,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),next.data_ptr<float>(),out.data_ptr<scalar_t>(),q.size(2),v.size(2),eps);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,next};
}
