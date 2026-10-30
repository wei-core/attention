#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <cmath>
#include <mma.h>
#include "../common/block_reduce.cuh"
namespace {
// A CTA cooperates on features and output columns; triangular rows remain ordered.
template<typename T,bool USE_MMA> __global__ void chunks(const T* q,const T* k,const T* v,const float* g,const float* beta,const float* initial,T* out,float* final,float* scratch,int Tlen,int H,int D,int V,int C,float eps) {
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
  // Blocked forward substitution: solve a small diagonal block for every
  // independent RHS column, then update the remaining rows in parallel.
  int width=D+V;
  for(int x=tid;x<n*width;x+=blockDim.x){int i=x/width,col=x%width;
   int64_t gh=(static_cast<int64_t>(b)*Tlen+start+i)*H+h;
   if(col<D)R[i*D+col]=beta[gh]*expf(G[i])*K[i*D+col];
   else U[i*V+col-D]=beta[gh]*float(v[gh*V+col-D]);
  }
  __syncthreads();
  constexpr int block_rows=8;
  for(int row0=0;row0<n;row0+=block_rows){
   int stop=min(n,row0+block_rows);
   for(int col=tid;col<width;col+=blockDim.x){
    for(int i=row0;i<stop;i++){
     float value=col<D?R[i*D+col]:U[i*V+col-D];
     for(int j=row0;j<i;j++)value-=A[i*C+j]*(col<D?R[j*D+col]:U[j*V+col-D]);
     if(col<D)R[i*D+col]=value;else U[i*V+col-D]=value;
    }
   }
   __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
   if constexpr(USE_MMA){
    // Two TF32 components recover near-FP32 precision for the block update.
    // Each warp owns one 16x16 output tile of A_remaining,block @ X_block.
    using namespace nvcuda;
    __shared__ __align__(32) float ah[4][128],al[4][128],bh[4][128],bl[4][128],prod[4][256];
    int warp=tid/32,lane=tid%32,mt=(n-stop+15)/16,nt=(width+15)/16;
    for(int tile=warp;tile<mt*nt;tile+=4){int mi=tile/nt,ni=tile%nt;
     for(int x=lane;x<128;x+=32){int r=x/8,k=x%8,i=stop+mi*16+r,j=row0+k;
      float av=(i<n && j<stop)?A[i*C+j]:0.f;float hi=wmma::__float_to_tf32(av);ah[warp][x]=hi;al[warp][x]=wmma::__float_to_tf32(av-hi);
      int bk=x/16,bc=x%16,col=ni*16+bc,row=row0+bk;float bv=(row<stop && col<width)?(col<D?R[row*D+col]:U[row*V+col-D]):0.f;
      hi=wmma::__float_to_tf32(bv);bh[warp][x]=hi;bl[warp][x]=wmma::__float_to_tf32(bv-hi);
     }
     __syncwarp();
     wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a0,a1;
     wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> b0,b1;
     wmma::fragment<wmma::accumulator,16,16,8,float> c;wmma::fill_fragment(c,0.f);
     wmma::load_matrix_sync(a0,ah[warp],8);wmma::load_matrix_sync(a1,al[warp],8);
     wmma::load_matrix_sync(b0,bh[warp],16);wmma::load_matrix_sync(b1,bl[warp],16);
     wmma::mma_sync(c,a0,b0,c);wmma::mma_sync(c,a1,b0,c);wmma::mma_sync(c,a0,b1,c);
     wmma::store_matrix_sync(prod[warp],c,16,wmma::mem_row_major);__syncwarp();
     for(int x=lane;x<256;x+=32){int i=stop+mi*16+x/16,col=ni*16+x%16;if(i<n && col<width){
      if(col<D)R[i*D+col]-=prod[warp][x];else U[i*V+col-D]-=prod[warp][x];}}
     __syncwarp();
    }
   } else
#endif
   {
   for(int x=tid;x<(n-stop)*width;x+=blockDim.x){int i=stop+x/width,col=x%width;
    float value=col<D?R[i*D+col]:U[i*V+col-D];
    for(int j=row0;j<stop;j++)value-=A[i*C+j]*(col<D?R[j*D+col]:U[j*V+col-D]);
    if(col<D)R[i*D+col]=value;else U[i*V+col-D]=value;
   }
   }
   __syncthreads();
  }
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
std::tuple<torch::Tensor,torch::Tensor> delta_chunkwise_cuda(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& g,const torch::Tensor& beta,const torch::Tensor& state,int64_t chunk,double eps,bool use_mma) {
 for(const auto& x:{q,k,v,g,beta,state})TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device(),"inputs must be contiguous on one CUDA device");
 TORCH_CHECK(q.dim()==4 && k.sizes()==q.sizes() && v.dim()==4 && q.size(0)==v.size(0) && q.size(1)==v.size(1) && q.size(2)==v.size(2),"q/k [B,T,H,Dk], v [B,T,H,Dv]");
 TORCH_CHECK(g.dim()==3 && beta.sizes()==g.sizes() && g.size(0)==q.size(0) && g.size(1)==q.size(1) && g.size(2)==q.size(2),"g/beta must be [B,T,H]");
 TORCH_CHECK(state.dim()==4 && state.size(0)==q.size(0) && state.size(1)==q.size(2) && state.size(2)==q.size(3) && state.size(3)==v.size(3) && q.size(2)>0 && q.size(3)>0 && v.size(3)>0,"invalid state dimensions");
 TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==v.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"invalid q/k/v dtype");
 TORCH_CHECK(g.scalar_type()==at::kFloat && beta.scalar_type()==at::kFloat && state.scalar_type()==at::kFloat,"g/beta/state must be float32");TORCH_CHECK(chunk==16 || chunk==32 || chunk==64,"chunk size must be 16, 32, or 64");TORCH_CHECK(std::isfinite(eps) && eps>0,"eps must be positive and finite");
 auto out=at::empty_like(v),final=at::empty_like(state);if(q.size(0)==0)return {out,final};
 c10::cuda::CUDAGuard guard(q.device());
 TORCH_CHECK(!use_mma || (chunk>=32 && q.size(3)+v.size(3)>=32 && at::cuda::getCurrentDeviceProperties()->major>=8),"WMMA requires chunk>=32, Dk+Dv>=32 and compute capability>=8.0");
 auto scratch=at::empty({q.size(0)*q.size(2),chunk*(1+3*q.size(3)+v.size(3)+chunk)},state.options());
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"delta_chunkwise",[&]{if(use_mma) chunks<scalar_t,true><<<q.size(0)*q.size(2),128,(chunk*q.size(3)<=8192?chunk*q.size(3)*sizeof(float):0),at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),out.data_ptr<scalar_t>(),final.data_ptr<float>(),scratch.data_ptr<float>(),q.size(1),q.size(2),q.size(3),v.size(3),chunk,eps);else chunks<scalar_t,false><<<q.size(0)*q.size(2),128,(chunk*q.size(3)<=8192?chunk*q.size(3)*sizeof(float):0),at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),out.data_ptr<scalar_t>(),final.data_ptr<float>(),scratch.data_ptr<float>(),q.size(1),q.size(2),q.size(3),v.size(3),chunk,eps);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,final};
}
