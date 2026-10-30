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
// Matrix absorption: Q W_K is formed once per query, and W_V is applied
// after the online-softmax weighted latent sum. FP32 shared intermediates.
template<typename T> __global__ void absorbed(const T* q,const T* qr,const T* latent,const T* rope,const T* wk,const T* wv,const int32_t* lengths,const int32_t* positions,T* out,int tq,int L,int H,int dn,int dr,int dc,int dv,float scale) {
 int h=blockIdx.x,b=blockIdx.y,t=blockIdx.z,tid=threadIdx.x;
 int end=min(lengths[b],max(0,positions[b*tq+t]+1));
 int64_t row=(static_cast<int64_t>(b)*tq+t)*H+h,qb=row*dn,rb=row*dr;
 extern __shared__ float shared[];float* qa=shared;float* acc=shared+dc;
 __shared__ float reduction[8];
 for(int c=tid;c<dc;c+=blockDim.x){float x=0;for(int d=0;d<dn;d++)x+=float(q[qb+d])*float(wk[(static_cast<int64_t>(h)*dn+d)*dc+c]);qa[c]=x;acc[c]=0;}
 __syncthreads();qd_attention::SoftmaxState state;
 for(int s=0;s<end;s++){
  int64_t cb=(static_cast<int64_t>(b)*L+s)*dc,kb=(static_cast<int64_t>(b)*L+s)*dr;
  float dot=0;for(int c=tid;c<dc;c+=blockDim.x)dot+=qa[c]*float(latent[cb+c]);
  for(int d=tid;d<dr;d+=blockDim.x)dot+=float(qr[rb+d])*float(rope[kb+d]);
  float score=qd_attention::block_sum(dot,reduction)*scale,a,p;state.add(score,a,p);
  for(int c=tid;c<dc;c+=blockDim.x)acc[c]=a*acc[c]+p*float(latent[cb+c]);
  __syncthreads();
 }
 for(int d=tid;d<dv;d+=blockDim.x){float x=0;for(int c=0;c<dc;c++)x+=float(wv[(static_cast<int64_t>(h)*dv+d)*dc+c])*acc[c];out[row*dv+d]=static_cast<T>(state.sum>0?x/state.sum:0.f);}
}

// Four query warps share each latent/rope row loaded into shared memory.
template<typename T> __global__ void absorbed_tiled(const T* q,const T* qr,const T* latent,const T* rope,const T* wk,const T* wv,const int32_t* lengths,const int32_t* positions,T* out,int tq,int L,int H,int dn,int dr,int dc,int dv,float scale){
 int h=blockIdx.x,b=blockIdx.y,t0=blockIdx.z*4,tid=threadIdx.x,warp=tid/32,lane=tid%32,t=t0+warp;
 extern __shared__ float sm[];float* qa=sm;float* acc=qa+4*dc;float* lat=acc+4*dc;float* kr=lat+4*dc;
 int end=t<tq?min(lengths[b],max(0,positions[b*tq+t]+1)):0;
 int maxend=0;for(int i=0;i<4;i++)if(t0+i<tq)maxend=max(maxend,min(lengths[b],max(0,positions[b*tq+t0+i]+1)));
 int64_t row=(static_cast<int64_t>(b)*tq+t)*H+h;
 for(int c=lane;c<dc;c+=32){float x=0;if(t<tq)for(int d=0;d<dn;d++)x+=float(q[row*dn+d])*float(wk[(static_cast<int64_t>(h)*dn+d)*dc+c]);qa[warp*dc+c]=x;acc[warp*dc+c]=0;}
 __syncthreads();qd_attention::SoftmaxState state;
 // Load four latent tokens per shared tile. All query warps reuse the tile.
 constexpr int tile_n=4;
 for(int base=0;base<maxend;base+=tile_n){
  int count=min(tile_n,maxend-base);
  for(int x=tid;x<count*dc;x+=blockDim.x)lat[x]=float(latent[(static_cast<int64_t>(b)*L+base+x/dc)*dc+x%dc]);
  for(int x=tid;x<count*dr;x+=blockDim.x)kr[x]=float(rope[(static_cast<int64_t>(b)*L+base+x/dr)*dr+x%dr]);
  __syncthreads();
  for(int j=0;j<count;j++)if(base+j<end){float dot=0;for(int c=lane;c<dc;c+=32)dot+=qa[warp*dc+c]*lat[j*dc+c];for(int d=lane;d<dr;d+=32)dot+=float(qr[row*dr+d])*kr[j*dr+d];
   for(int off=16;off;off>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,off);dot=__shfl_sync(0xffffffffu,dot,0);
   float a,p;state.add(dot*scale,a,p);for(int c=lane;c<dc;c+=32)acc[warp*dc+c]=a*acc[warp*dc+c]+p*lat[j*dc+c];}
  __syncthreads();
 }
 if(t<tq)for(int d=lane;d<dv;d+=32){float x=0;for(int c=0;c<dc;c++)x+=float(wv[(static_cast<int64_t>(h)*dv+d)*dc+c])*acc[warp*dc+c];out[row*dv+d]=static_cast<T>(state.sum>0?x/state.sum:0.f);}
}

template<typename T> __device__ float to_float(T x);
template<> __device__ float to_float<__half>(__half x){return __half2float(x);}
template<> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x){return __bfloat162float(x);}
template<typename T> __device__ T from_float(float x);
template<> __device__ __half from_float<__half>(float x){return __float2half_rn(x);}
template<> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x){return __float2bfloat16_rn(x);}
// Tensor Core path: 16 query rows and 16 latent tokens per tile.
// Q absorption, online state and value projection accumulate in FP32;
// WMMA inputs and softmax probabilities are converted to input precision.
template<typename T> __global__ void absorbed_wmma(const T* q,const T* qr,const T* latent,const T* rope,const T* wk,const T* wv,const int32_t* lengths,const int32_t* positions,T* out,int tq,int L,int H,int dn,int dr,int dc,int dv,float scale){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 using namespace nvcuda;int h=blockIdx.x,b=blockIdx.y,t0=blockIdx.z*16,lane=threadIdx.x;
 __shared__ __align__(32) T qa[16*256],a[256],bb[256],prob[256];
 __shared__ __align__(32) float acc[16*256],tile[256];__shared__ float mx[16],den[16],alpha[16];
 for(int x=lane;x<16*dc;x+=32){int t=t0+x/dc,c=x%dc;float v=0;if(t<tq){int64_t row=(static_cast<int64_t>(b)*tq+t)*H+h;for(int d=0;d<dn;d++)v+=to_float(q[row*dn+d])*to_float(wk[(static_cast<int64_t>(h)*dn+d)*dc+c]);}qa[x]=from_float<T>(v);acc[x]=0;}
 if(lane<16){mx[lane]=-CUDART_INF_F;den[lane]=0;}__syncwarp();
 int maxend=0;for(int i=0;i<16;i++)if(t0+i<tq)maxend=max(maxend,min(lengths[b],max(0,positions[b*tq+t0+i]+1)));
 for(int start=0;start<maxend;start+=16){
  wmma::fragment<wmma::accumulator,16,16,16,float> score;wmma::fill_fragment(score,0.f);
  for(int ds=0;ds<dc+dr;ds+=16){
   for(int x=lane;x<256;x+=32){int r=x/16,c=x%16,t=t0+r,s=start+r,d=ds+c;
    if(d<dc){a[x]=qa[r*dc+d];bb[x]=s<maxend?latent[(static_cast<int64_t>(b)*L+s)*dc+d]:from_float<T>(0.f);}
    else {int rd=d-dc;a[x]=(t<tq && rd<dr)?qr[((static_cast<int64_t>(b)*tq+t)*H+h)*dr+rd]:from_float<T>(0.f);bb[x]=(s<maxend && rd<dr)?rope[(static_cast<int64_t>(b)*L+s)*dr+rd]:from_float<T>(0.f);}
   }
   __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> av;wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bv;
   wmma::load_matrix_sync(av,a,16);wmma::load_matrix_sync(bv,bb,16);wmma::mma_sync(score,av,bv,score);__syncwarp();
  }
  wmma::store_matrix_sync(tile,score,16,wmma::mem_row_major);__syncwarp();
  if(lane<16){int t=t0+lane,end=t<tq?min(lengths[b],max(0,positions[b*tq+t]+1)):0;float m=mx[lane];
   for(int j=0;j<16;j++)if(start+j<end)m=fmaxf(m,tile[lane*16+j]*scale);
   float old=den[lane]>0?expf(mx[lane]-m):0.f,sum=0;
   for(int j=0;j<16;j++){float p=(start+j<end)?expf(tile[lane*16+j]*scale-m):0.f;prob[lane*16+j]=from_float<T>(p);sum+=p;}
   alpha[lane]=old;den[lane]=den[lane]*old+sum;mx[lane]=m;
  }
  __syncwarp();for(int x=lane;x<16*dc;x+=32)acc[x]*=alpha[x/dc];__syncwarp();
  for(int ds=0;ds<dc;ds+=16){
   for(int x=lane;x<256;x+=32){int s=start+x/16,d=ds+x%16;bb[x]=(s<maxend && d<dc)?latent[(static_cast<int64_t>(b)*L+s)*dc+d]:from_float<T>(0.f);}
   __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> pv;wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> lv;wmma::fragment<wmma::accumulator,16,16,16,float> result;
   wmma::fill_fragment(result,0.f);wmma::load_matrix_sync(pv,prob,16);wmma::load_matrix_sync(lv,bb,16);wmma::mma_sync(result,pv,lv,result);wmma::store_matrix_sync(tile,result,16,wmma::mem_row_major);__syncwarp();
   for(int x=lane;x<256;x+=32){int d=ds+x%16;if(d<dc)acc[(x/16)*dc+d]+=tile[x];}__syncwarp();
  }
 }
 for(int x=lane;x<16*dv;x+=32){int r=x/dv,t=t0+r,d=x%dv;if(t<tq){float v=0;for(int c=0;c<dc;c++)v+=to_float(wv[(static_cast<int64_t>(h)*dv+d)*dc+c])*acc[r*dc+c];out[((static_cast<int64_t>(b)*tq+t)*H+h)*dv+d]=from_float<T>(den[r]>0?v/den[r]:0.f);}}
#endif
}

}
torch::Tensor mla_prefill_direct_cuda(const torch::Tensor& q,const torch::Tensor& qr,const torch::Tensor& latent,const torch::Tensor& rope,const torch::Tensor& wk,const torch::Tensor& wv,const torch::Tensor& lens,const torch::Tensor& positions,double scale,bool use_mma) {
 for(const auto& x:{q,qr,latent,rope,wk,wv,lens,positions})TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device(),"all inputs must be contiguous on one CUDA device");
 TORCH_CHECK(q.dim()==4 && qr.dim()==4 && latent.dim()==3 && rope.dim()==3 && wk.dim()==3 && wv.dim()==3 && lens.dim()==1 && positions.dim()==2,"invalid tensor ranks");
 TORCH_CHECK(qr.size(0)==q.size(0) && qr.size(1)==q.size(1) && qr.size(2)==q.size(2) && latent.size(0)==q.size(0) && rope.size(0)==q.size(0) && rope.size(1)==latent.size(1) && rope.size(2)==qr.size(3),"invalid input shapes");
 TORCH_CHECK(wk.size(0)==q.size(2) && wk.size(1)==q.size(3) && wk.size(2)==latent.size(2) && wv.size(0)==q.size(2) && wv.size(2)==latent.size(2),"invalid projection shapes");
 TORCH_CHECK(q.size(2)>0 && q.size(3)>0 && qr.size(3)>0 && latent.size(2)>0 && wv.size(1)>0 && wv.size(1)<=512 && q.size(0)<=65535 && q.size(1)<=65535,"unsupported dimensions");
 TORCH_CHECK(lens.size(0)==q.size(0) && positions.size(0)==q.size(0) && positions.size(1)==q.size(1) && lens.scalar_type()==at::kInt && positions.scalar_type()==at::kInt,"invalid metadata");
 TORCH_CHECK(q.scalar_type()==qr.scalar_type() && q.scalar_type()==latent.scalar_type() && q.scalar_type()==rope.scalar_type() && q.scalar_type()==wk.scalar_type() && q.scalar_type()==wv.scalar_type() && (q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"dtype mismatch or unsupported");TORCH_CHECK(std::isfinite(scale) && scale>0,"invalid scale");
 auto out=at::empty({q.size(0),q.size(1),q.size(2),wv.size(1)},q.options());if(!out.numel())return out;c10::cuda::CUDAGuard guard(q.device());dim3 grid(q.size(2),q.size(0),q.size(1));
 TORCH_CHECK(!use_mma || (q.size(1)>=16 && latent.size(2)<=256 && (q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16) && at::cuda::getCurrentDeviceProperties()->major>=8),"WMMA requires T>=16, Dc<=256, FP16/BF16, compute capability>=8.0");
 if(use_mma){
  dim3 tiles(q.size(2),q.size(0),(q.size(1)+15)/16);auto stream=at::cuda::getCurrentCUDAStream().stream();
  if(q.scalar_type()==at::kHalf)absorbed_wmma<__half><<<tiles,32,0,stream>>>(reinterpret_cast<const __half*>(q.data_ptr()),reinterpret_cast<const __half*>(qr.data_ptr()),reinterpret_cast<const __half*>(latent.data_ptr()),reinterpret_cast<const __half*>(rope.data_ptr()),reinterpret_cast<const __half*>(wk.data_ptr()),reinterpret_cast<const __half*>(wv.data_ptr()),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),reinterpret_cast<__half*>(out.data_ptr()),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);
  else absorbed_wmma<__nv_bfloat16><<<tiles,32,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(latent.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(rope.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(wk.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(wv.data_ptr()),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
 }
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"mla_prefill_direct",[&]{if(q.size(1)>=8 && (12*latent.size(2)+4*qr.size(3))*sizeof(float)<=32768){dim3 tiles(q.size(2),q.size(0),(q.size(1)+3)/4);absorbed_tiled<scalar_t><<<tiles,128,(12*latent.size(2)+4*qr.size(3))*sizeof(float),at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),qr.data_ptr<scalar_t>(),latent.data_ptr<scalar_t>(),rope.data_ptr<scalar_t>(),wk.data_ptr<scalar_t>(),wv.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<scalar_t>(),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);}else if(latent.size(2)<=4096) absorbed<scalar_t><<<grid,256,2*latent.size(2)*sizeof(float),at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),qr.data_ptr<scalar_t>(),latent.data_ptr<scalar_t>(),rope.data_ptr<scalar_t>(),wk.data_ptr<scalar_t>(),wv.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<scalar_t>(),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);else direct<scalar_t><<<grid,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),qr.data_ptr<scalar_t>(),latent.data_ptr<scalar_t>(),rope.data_ptr<scalar_t>(),wk.data_ptr<scalar_t>(),wv.data_ptr<scalar_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),out.data_ptr<scalar_t>(),q.size(1),latent.size(1),q.size(2),q.size(3),qr.size(3),latent.size(2),wv.size(1),scale);});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
