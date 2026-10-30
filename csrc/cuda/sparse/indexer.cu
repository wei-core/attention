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
 // Each warp computes one score; lanes cooperate on feature reduction.
 int64_t i=static_cast<int64_t>(blockIdx.x)*4+(threadIdx.x>>5);
 int lane=threadIdx.x&31; int64_t n=static_cast<int64_t>(bsz)*tq*l;
 if(i>=n)return;
 int s=i%l,t=(i/l)%tq,b=i/(static_cast<int64_t>(l)*tq);
 if(s>=lens[b] || s>positions[b*tq+t]) {if(!lane)out[i]=-CUDART_INF_F;return;}
 float score=0;
 for(int head=0;head<h;head++) {
  float dot=0;for(int j=lane;j<d;j+=32)dot+=float(q[((static_cast<int64_t>(b)*tq+t)*h+head)*d+j])*float(k[(static_cast<int64_t>(b)*l+s)*d+j]);
  for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
  if(!lane)score+=float(w[(static_cast<int64_t>(b)*tq+t)*h+head])*fmaxf(dot,0.f);
 }
 if(!lane)out[i]=score;
}
__global__ void topk_kernel(const float* scores,int32_t* out,int rows,int l,int k) {
 // Parallel row scan and deterministic score/index reduction on each selection.
 int row=blockIdx.x, tid=threadIdx.x;
 __shared__ float vals[128], prev_score;
 __shared__ int inds[128], prev_idx;
 if(!tid){prev_score=CUDART_INF_F;prev_idx=-1;}
 __syncthreads();
 for(int j=0;j<k;j++) {
  float best=-CUDART_INF_F;int idx=-1;
  for(int i=tid;i<l;i+=blockDim.x) {
   float x=scores[static_cast<int64_t>(row)*l+i];
   bool after=x<prev_score || (x==prev_score && i>prev_idx);
   if(isfinite(x) && after && (idx<0 || x>best || (x==best && i<idx))){best=x;idx=i;}
  }
  vals[tid]=best;inds[tid]=idx;__syncthreads();
  for(int step=blockDim.x/2;step;step>>=1){
   if(tid<step){float x=vals[tid+step];int i=inds[tid+step];
    if(i>=0 && (inds[tid]<0 || x>vals[tid] || (x==vals[tid] && i<inds[tid]))){vals[tid]=x;inds[tid]=i;}}
   __syncthreads();
  }
  if(!tid){int i=inds[0];out[static_cast<int64_t>(row)*k+j]=i;if(i>=0){prev_score=vals[0];prev_idx=i;}}
  __syncthreads();
 }
}
// Select up to K candidates independently from each 256-element tile.
__global__ void topk_tiles(const float* scores,int32_t* candidates,int l,int k,int tiles) {
 int row=blockIdx.x,tile=blockIdx.y,tid=threadIdx.x;
 __shared__ float vals[128],prev_score;__shared__ int inds[128],prev_idx;
 if(!tid){prev_score=CUDART_INF_F;prev_idx=-1;}__syncthreads();
 for(int j=0;j<k;j++){
  float best=-CUDART_INF_F;int idx=-1;
  for(int i=tile*256+tid;i<min(l,(tile+1)*256);i+=128){float x=scores[static_cast<int64_t>(row)*l+i];
   if(isfinite(x) && (x<prev_score || (x==prev_score && i>prev_idx)) && (idx<0 || x>best || (x==best && i<idx))){best=x;idx=i;}}
  vals[tid]=best;inds[tid]=idx;__syncthreads();
  for(int step=64;step;step>>=1){if(tid<step){int i=inds[tid+step];float x=vals[tid+step];if(i>=0 && (inds[tid]<0 || x>vals[tid] || (x==vals[tid] && i<inds[tid]))){vals[tid]=x;inds[tid]=i;}}__syncthreads();}
  if(!tid){int i=inds[0];candidates[(static_cast<int64_t>(row)*tiles+tile)*k+j]=i;if(i>=0){prev_score=vals[0];prev_idx=i;}}__syncthreads();
 }
}
__global__ void topk_merge(const float* scores,const int32_t* candidates,int32_t* out,int l,int k,int tiles){
 int row=blockIdx.x,tid=threadIdx.x;__shared__ float vals[128],prev_score;__shared__ int inds[128],prev_idx;
 if(!tid){prev_score=CUDART_INF_F;prev_idx=-1;}__syncthreads();
 for(int j=0;j<k;j++){
  float best=-CUDART_INF_F;int idx=-1;
  for(int c=tid;c<tiles*k;c+=128){int i=candidates[static_cast<int64_t>(row)*tiles*k+c];if(i<0)continue;float x=scores[static_cast<int64_t>(row)*l+i];
   if((x<prev_score || (x==prev_score && i>prev_idx)) && (idx<0 || x>best || (x==best && i<idx))){best=x;idx=i;}}
  vals[tid]=best;inds[tid]=idx;__syncthreads();
  for(int step=64;step;step>>=1){if(tid<step){int i=inds[tid+step];float x=vals[tid+step];if(i>=0 && (inds[tid]<0 || x>vals[tid] || (x==vals[tid] && i<inds[tid]))){vals[tid]=x;inds[tid]=i;}}__syncthreads();}
  if(!tid){int i=inds[0];out[static_cast<int64_t>(row)*k+j]=i;if(i>=0){prev_score=vals[0];prev_idx=i;}}__syncthreads();
 }
}

}
torch::Tensor lightning_indexer_cuda(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& w,const torch::Tensor& lens,const torch::Tensor& positions) {
 for(const auto& x:{q,k,w,lens,positions})check(x,q);
 TORCH_CHECK(q.dim()==4 && k.dim()==3 && w.dim()==3 && lens.dim()==1 && positions.dim()==2,"q [B,T,H,D], k [B,L,D], weights [B,T,H]");
 TORCH_CHECK(q.size(0)==k.size(0) && q.size(3)==k.size(2) && w.size(0)==q.size(0) && w.size(1)==q.size(1) && w.size(2)==q.size(2) && lens.size(0)==q.size(0) && positions.size(0)==q.size(0) && positions.size(1)==q.size(1),"shape mismatch");
 TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==w.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"dtype mismatch or unsupported");TORCH_CHECK(lens.scalar_type()==at::kInt && positions.scalar_type()==at::kInt,"metadata must be int32");
 auto out=at::empty({q.size(0),q.size(1),k.size(1)},q.options().dtype(at::kFloat));if(!out.numel())return out;
 c10::cuda::CUDAGuard guard(q.device());dim3 grid((out.numel()+3)/4);
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"lightning_indexer",[&]{index_kernel<scalar_t><<<grid,128,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<float>(),q.size(0),q.size(1),k.size(1),q.size(2),q.size(3));});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor topk_select_cuda(const torch::Tensor& scores,int64_t k) {
 TORCH_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim()==3 && scores.scalar_type()==at::kFloat,"scores must be contiguous CUDA float32 [B,T,L]");TORCH_CHECK(k>=0 && k<=std::numeric_limits<int>::max(),"invalid k");
 auto out=at::empty({scores.size(0),scores.size(1),k},scores.options().dtype(at::kInt));if(!out.numel())return out;
 c10::cuda::CUDAGuard guard(scores.device());int rows=scores.size(0)*scores.size(1);int l=scores.size(2);auto stream=at::cuda::getCurrentCUDAStream().stream();
 if(l>256 && k<=64 && (l+255)/256<=65535){int tiles=(l+255)/256;auto candidates=at::empty({rows,tiles,k},scores.options().dtype(at::kInt));topk_tiles<<<dim3(rows,tiles),128,0,stream>>>(scores.data_ptr<float>(),candidates.data_ptr<int32_t>(),l,k,tiles);C10_CUDA_KERNEL_LAUNCH_CHECK();topk_merge<<<rows,128,0,stream>>>(scores.data_ptr<float>(),candidates.data_ptr<int32_t>(),out.data_ptr<int32_t>(),l,k,tiles);}
 else topk_kernel<<<rows,128,0,stream>>>(scores.data_ptr<float>(),out.data_ptr<int32_t>(),rows,l,k);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
