#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include "../common/block_reduce.cuh"
namespace {
// A CTA cooperates on features and output columns; triangular rows remain ordered.
template<typename T> __global__ void chunks(const T* q,const T* k,const T* v,const float* g,const float* beta,const float* initial,T* out,float* final,float* scratch,int Tlen,int H,int D,int V,int C,float eps) {
 int bh=blockIdx.x,b=bh/H,h=bh%H,tid=threadIdx.x;
 int64_t sb=static_cast<int64_t>(bh)*D*V, stride=static_cast<int64_t>(C)*(1+3*D+V+C);
 float* G=scratch+bh*stride;float* Q=G+C;float* K=Q+C*D;float* R=K+C*D;float* U=R+C*D;float* A=U+C*V;
 extern __shared__ float kshared[];
 __shared__ float red[8], shared[3];
 for(int x=tid;x<D*V;x+=blockDim.x)final[sb+x]=initial[sb+x];
 __syncthreads();
 for(int start=0;start<Tlen;start+=C){
  int n=min(C,Tlen-start);
  if(!tid){float sum=0;for(int i=0;i<n;i++){sum+=g[(static_cast<int64_t>(b)*Tlen+start+i)*H+h];G[i]=sum;}}
  __syncthreads();
  for(int i=0;i<n;i++){
   int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;float qn=0,kn=0;
   for(int d=tid;d<D;d+=blockDim.x){float x=float(q[gh*D+d]),y=float(k[gh*D+d]);qn+=x*x;kn+=y*y;}
   qn=qd_attention::block_sum(qn,red);kn=qd_attention::block_sum(kn,red);
   if(!tid){shared[0]=rsqrtf(qn+eps)*rsqrtf(float(D));shared[1]=rsqrtf(kn+eps);}
   __syncthreads();
   for(int d=tid;d<D;d+=blockDim.x){Q[i*D+d]=float(q[gh*D+d])*shared[0];K[i*D+d]=float(k[gh*D+d])*shared[1];}
   __syncthreads();
  }
  bool cached=C*D<=8192;
  if(cached)for(int x=tid;x<n*D;x+=blockDim.x)kshared[x]=K[x];
  __syncthreads();
  // Construct the lower triangular Gram tile in parallel. K is cached
  // in shared memory when its footprint fits the launch budget.
  for(int x=tid;x<n*n;x+=blockDim.x){int i=x/n,j=x%n;float dot=0;
   if(j<i){for(int d=0;d<D;d++)dot+=(cached?kshared[i*D+d]:K[i*D+d])*(cached?kshared[j*D+d]:K[j*D+d]);
    int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;A[i*C+j]=beta[gh]*dot*expf(G[i]-G[j]);}
  }
  __syncthreads();
  // Independent RHS columns solve the same lower triangular system in parallel.
  for(int col=tid;col<D+V;col+=blockDim.x){
   for(int i=0;i<n;i++){
    int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;
    float value=col<D?beta[gh]*expf(G[i])*K[i*D+col]:beta[gh]*float(v[gh*V+col-D]);
    for(int j=0;j<i;j++)value-=A[i*C+j]*(col<D?R[j*D+col]:U[j*V+col-D]);
    if(col<D)R[i*D+col]=value;else U[i*V+col-D]=value;
   }
  }
  __syncthreads();
  for(int x=tid;x<n*V;x+=blockDim.x){int i=x/V,c=x%V;float p=0;for(int d=0;d<D;d++)p+=R[i*D+d]*final[sb+d*V+c];U[x]-=p;}
  __syncthreads();
  for(int x=tid;x<n*V;x+=blockDim.x){int i=x/V,c=x%V;float value=0;for(int d=0;d<D;d++)value+=expf(G[i])*Q[i*D+d]*final[sb+d*V+c];
   for(int j=0;j<=i;j++){float dot=0;for(int d=0;d<D;d++)dot+=Q[i*D+d]*K[j*D+d];value+=expf(G[i]-G[j])*dot*U[j*V+c];}
   out[((static_cast<int64_t>(b)*Tlen+start+i)*H+h)*V+c]=static_cast<T>(value);
  }
  __syncthreads();
  for(int x=tid;x<D*V;x+=blockDim.x){int d=x/V,c=x%V;float value=expf(G[n-1])*final[sb+x];for(int j=0;j<n;j++)value+=expf(G[n-1]-G[j])*K[j*D+d]*U[j*V+c];final[sb+x]=value;}
  __syncthreads();
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
 c10::cuda::CUDAGuard guard(q.device());auto scratch=at::empty({q.size(0)*q.size(2),chunk*(1+3*q.size(3)+v.size(3)+chunk)},state.options());
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"delta_chunkwise",[&]{chunks<scalar_t><<<q.size(0)*q.size(2),128,(chunk*q.size(3)<=8192?chunk*q.size(3)*sizeof(float):0),at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),out.data_ptr<scalar_t>(),final.data_ptr<float>(),scratch.data_ptr<float>(),q.size(1),q.size(2),q.size(3),v.size(3),chunk,eps);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,final};
}
