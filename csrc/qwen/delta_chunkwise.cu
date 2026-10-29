#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
namespace {
// Correctness-first implementation: one thread per head, FP32 scratch and state.
// Explicit forward substitution solves the unit-lower-triangular chunk system.
template<typename T> __global__ void chunks(const T* q,const T* k,const T* v,const float* g,const float* beta,const float* initial,T* out,float* final,float* scratch,int Tlen,int H,int D,int V,int C,float eps) {
 int bh=blockIdx.x,b=bh/H,h=bh%H; if(threadIdx.x) return;
 int64_t sb=static_cast<int64_t>(bh)*D*V;
 int64_t stride=static_cast<int64_t>(C)*(1+3*D+V);
 float* G=scratch+bh*stride;float* Q=G+C;float* K=Q+C*D;float* R=K+C*D;float* U=R+C*D;
 for(int j=0;j<D*V;j++) final[sb+j]=initial[sb+j];
 for(int start=0;start<Tlen;start+=C) {
  int n=min(C,Tlen-start);float cumulative=0;
  for(int i=0;i<n;i++) {
   int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;
   float qnorm=0,knorm=0;
   for(int d=0;d<D;d++){float x=float(q[gh*D+d]),y=float(k[gh*D+d]);qnorm+=x*x;knorm+=y*y;}
   float qs=rsqrtf(qnorm+eps)*rsqrtf(float(D)),ks=rsqrtf(knorm+eps);
   for(int d=0;d<D;d++){Q[i*D+d]=float(q[gh*D+d])*qs;K[i*D+d]=float(k[gh*D+d])*ks;}
   cumulative+=g[gh];G[i]=cumulative;
  }
  for(int i=0;i<n;i++) {
   int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;float bt=beta[gh];
   for(int d=0;d<D;d++)R[i*D+d]=bt*expf(G[i])*K[i*D+d];
   for(int col=0;col<V;col++)U[i*V+col]=bt*float(v[gh*V+col]);
   for(int j=0;j<i;j++) {
    float dot=0;for(int d=0;d<D;d++)dot+=K[i*D+d]*K[j*D+d];float a=bt*dot*expf(G[i]-G[j]);
    for(int d=0;d<D;d++)R[i*D+d]-=a*R[j*D+d];
    for(int col=0;col<V;col++)U[i*V+col]-=a*U[j*V+col];
   }
  }
  for(int i=0;i<n;i++)for(int col=0;col<V;col++){float p=0;for(int d=0;d<D;d++)p+=R[i*D+d]*final[sb+d*V+col];U[i*V+col]-=p;}
  for(int i=0;i<n;i++)for(int col=0;col<V;col++) {
   float value=0;for(int d=0;d<D;d++)value+=expf(G[i])*Q[i*D+d]*final[sb+d*V+col];
   for(int j=0;j<=i;j++){float dot=0;for(int d=0;d<D;d++)dot+=Q[i*D+d]*K[j*D+d];value+=expf(G[i]-G[j])*dot*U[j*V+col];}
   out[((static_cast<int64_t>(b)*Tlen+start+i)*H+h)*V+col]=static_cast<T>(value);
  }
  for(int d=0;d<D;d++)for(int col=0;col<V;col++){float value=expf(G[n-1])*final[sb+d*V+col];for(int j=0;j<n;j++)value+=expf(G[n-1]-G[j])*K[j*D+d]*U[j*V+col];final[sb+d*V+col]=value;}
 }
}
}
std::tuple<torch::Tensor,torch::Tensor> delta_chunkwise_cuda(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& g,const torch::Tensor& beta,const torch::Tensor& state,int64_t chunk,double eps) {
 for(const auto& x:{q,k,v,g,beta,state})TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device(),"inputs must be contiguous on one CUDA device");
 TORCH_CHECK(q.dim()==4 && k.sizes()==q.sizes() && v.dim()==4 && q.size(0)==v.size(0) && q.size(1)==v.size(1) && q.size(2)==v.size(2),"q/k [B,T,H,Dk], v [B,T,H,Dv]");
 TORCH_CHECK(g.dim()==3 && beta.sizes()==g.sizes() && g.size(0)==q.size(0) && g.size(1)==q.size(1) && g.size(2)==q.size(2),"g/beta must be [B,T,H]");
 TORCH_CHECK(state.dim()==4 && state.size(0)==q.size(0) && state.size(1)==q.size(2) && state.size(2)==q.size(3) && state.size(3)==v.size(3) && q.size(2)>0 && q.size(3)>0 && v.size(3)>0,"invalid state dimensions");
 TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==v.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"invalid q/k/v dtype");
 TORCH_CHECK(g.scalar_type()==at::kFloat && beta.scalar_type()==at::kFloat && state.scalar_type()==at::kFloat,"g/beta/state must be float32");TORCH_CHECK(chunk==16 || chunk==32 || chunk==64,"chunk size must be 16, 32, or 64");TORCH_CHECK(std::isfinite(eps) && eps>0,"eps must be positive and finite");
 auto out=at::empty_like(v),final=at::empty_like(state);if(q.size(0)==0)return {out,final};
 c10::cuda::CUDAGuard guard(q.device());auto scratch=at::empty({q.size(0)*q.size(2),chunk*(1+3*q.size(3)+v.size(3))},state.options());
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"delta_chunkwise",[&]{chunks<scalar_t><<<q.size(0)*q.size(2),1,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),out.data_ptr<scalar_t>(),final.data_ptr<float>(),scratch.data_ptr<float>(),q.size(1),q.size(2),q.size(3),v.size(3),chunk,eps);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,final};
}
