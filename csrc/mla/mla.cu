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
void check(const torch::Tensor& x, const torch::Tensor& base, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device() == base.device(), name, " must be contiguous and on the same CUDA device");
}
void dtype(const torch::Tensor& x) { TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16, "dtype must be float32/float16/bfloat16"); }
template <typename T, bool ABSORB>
__global__ void projection(const T* x, const T* w, T* out, int rows, int heads, int di, int dout) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<int64_t>(rows)*heads*dout) return;
    int col=i%dout, h=(i/dout)%heads; int64_t row=i/(heads*dout);
    float sum=0;
    for(int j=0;j<di;j++) {
        int64_t wi=ABSORB ? (static_cast<int64_t>(h)*di+j)*dout+col : (static_cast<int64_t>(h)*dout+col)*di+j;
        sum+=static_cast<float>(x[(row*heads+h)*di+j])*static_cast<float>(w[wi]);
    }
    out[i]=static_cast<T>(sum);
}
template <typename T>
__global__ void write_cache(const T* latent,const T* rope,T* cache,const int64_t* slots,int n,int dc,int dr) {
    int t=blockIdx.x;
    for(int d=threadIdx.x;d<dc+dr;d+=blockDim.x) cache[slots[t]*(dc+dr)+d]=d<dc?latent[static_cast<int64_t>(t)*dc+d]:rope[static_cast<int64_t>(t)*dr+d-dc];
}
template <typename T>
__global__ void attend(const T* q,const T* qr,const T* cache,const int32_t* table,const int32_t* lens,const int32_t* pos,const int32_t* selected,T* out,
    int tq,int hq,int dc,int dr,int page,int pages,int nk,float scale,bool causal,bool sparse) {
    int h=blockIdx.x,b=blockIdx.y,t=blockIdx.z,tid=threadIdx.x;
    int len=lens[b], end=causal?min(len,max(0,pos[b*tq+t]+1)):len;
    int count=sparse?nk:end;
    int64_t qb=((static_cast<int64_t>(b)*tq+t)*hq+h)*dc;
    int64_t rb=((static_cast<int64_t>(b)*tq+t)*hq+h)*dr;
    __shared__ float scratch[8]; qd_attention::SoftmaxState state; float acc[2]={0.f,0.f};
    for(int i=0;i<count;i++) {
        int s=sparse?selected[(static_cast<int64_t>(b)*tq+t)*nk+i]:i;
        if(s<0 || s>=end) continue;
        int phys=table[static_cast<int64_t>(b)*pages+s/page];
        int64_t base=(static_cast<int64_t>(phys)*page+s%page)*(dc+dr);
        float dot=0;
        for(int d=tid;d<dc;d+=blockDim.x) dot+=float(q[qb+d])*float(cache[base+d]);
        for(int d=tid;d<dr;d+=blockDim.x) dot+=float(qr[rb+d])*float(cache[base+dc+d]);
        float score=qd_attention::block_sum(dot,scratch)*scale, a,p;
        state.add(score,a,p);
        for(int d=tid;d<dc;d+=blockDim.x) acc[d/blockDim.x]=a*acc[d/blockDim.x]+p*float(cache[base+d]);
    }
    for(int d=tid;d<dc;d+=blockDim.x) out[qb+d]=static_cast<T>(state.sum>0?acc[d/blockDim.x]/state.sum:0.f);
}
}

torch::Tensor mla_project_cuda(const torch::Tensor& x,const torch::Tensor& w,bool absorb) {
    check(x,x,"x"); check(w,x,"w"); dtype(x); TORCH_CHECK(x.scalar_type()==w.scalar_type(),"dtype mismatch");
    TORCH_CHECK(x.dim()==4 && w.dim()==3 && x.size(2)==w.size(0),"x [B,T,H,D], weight [H,Dn,Dc] or [H,Dv,Dc]");
    int64_t di=x.size(3), dout=absorb?w.size(2):w.size(1);
    TORCH_CHECK(di>0 && dout>0 && (absorb?w.size(1)==di:w.size(2)==di),"projection dimension mismatch");
    auto out=at::empty({x.size(0),x.size(1),x.size(2),dout},x.options());
    if(out.numel()==0) return out;
    c10::cuda::CUDAGuard guard(x.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"mla_project",[&] {
      if(absorb) projection<scalar_t,true><<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(x.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),x.size(0)*x.size(1),x.size(2),di,dout);
      else projection<scalar_t,false><<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(x.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),x.size(0)*x.size(1),x.size(2),di,dout);
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
void latent_kv_write_cuda(const torch::Tensor& latent,const torch::Tensor& rope,torch::Tensor cache,const torch::Tensor& slots) {
    check(latent,latent,"latent"); check(rope,latent,"rope"); check(cache,latent,"cache"); check(slots,latent,"slots"); dtype(latent);
    TORCH_CHECK(latent.dim()==2 && rope.dim()==2 && cache.dim()==3 && slots.dim()==1 && latent.size(0)==rope.size(0) && slots.size(0)==latent.size(0),"invalid latent/rope/cache/slots shape");
    TORCH_CHECK(latent.size(1)>0 && rope.size(1)>0 && cache.size(1)>0 && cache.size(2)==latent.size(1)+rope.size(1),"invalid cache width");
    TORCH_CHECK(slots.scalar_type()==at::kLong && latent.scalar_type()==rope.scalar_type() && latent.scalar_type()==cache.scalar_type(),"slots must be int64 and tensors must share dtype");
    if(!latent.size(0)) return;
    c10::cuda::CUDAGuard guard(latent.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,latent.scalar_type(),"latent_write",[&] { write_cache<scalar_t><<<latent.size(0),256,0,at::cuda::getCurrentCUDAStream().stream()>>>(latent.data_ptr<scalar_t>(),rope.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),slots.data_ptr<int64_t>(),latent.size(0),latent.size(1),rope.size(1)); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor mla_attend_cuda(const torch::Tensor& q,const torch::Tensor& qr,const torch::Tensor& cache,const torch::Tensor& table,const torch::Tensor& lens,const torch::Tensor& positions,const c10::optional<torch::Tensor>& indices,double scale,bool causal) {
    check(q,q,"q"); check(qr,q,"q_rope"); check(cache,q,"cache"); check(table,q,"table"); check(lens,q,"lengths"); check(positions,q,"positions"); dtype(q);
    TORCH_CHECK(q.dim()==4 && qr.dim()==4 && cache.dim()==3 && table.dim()==2 && lens.dim()==1 && positions.dim()==2,"invalid tensor ranks");
    TORCH_CHECK(q.size(0)==qr.size(0) && q.size(1)==qr.size(1) && q.size(2)==qr.size(2) && q.size(0)==table.size(0) && q.size(0)==lens.size(0) && positions.size(0)==q.size(0) && positions.size(1)==q.size(1),"shape mismatch");
    TORCH_CHECK(q.size(3)>0 && q.size(3)<=512 && qr.size(3)>0 && cache.size(2)==q.size(3)+qr.size(3) && cache.size(0)>0 && cache.size(1)>0 && table.size(1)>0 && q.size(2)>0,"invalid dimensions");
    TORCH_CHECK(q.size(0)<=65535 && q.size(1)<=65535,"CUDA grid limit");
    TORCH_CHECK(q.scalar_type()==qr.scalar_type() && q.scalar_type()==cache.scalar_type(),"dtype mismatch");
    TORCH_CHECK(table.scalar_type()==at::kInt && lens.scalar_type()==at::kInt && positions.scalar_type()==at::kInt,"metadata must be int32");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"scale must be positive and finite");
    int nk=0; const int32_t* ptr=nullptr;
    if(indices.has_value()) { auto ix=*indices; check(ix,q,"indices"); TORCH_CHECK(ix.dim()==3 && ix.size(0)==q.size(0) && ix.size(1)==q.size(1) && ix.scalar_type()==at::kInt,"indices must be int32 [B,T,K]"); nk=ix.size(2); ptr=ix.data_ptr<int32_t>(); }
    auto out=at::empty_like(q); if(q.numel()==0) return out;
    c10::cuda::CUDAGuard guard(q.device()); dim3 grid(q.size(2),q.size(0),q.size(1));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,q.scalar_type(),"mla_attend",[&] { attend<scalar_t><<<grid,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<scalar_t>(),qr.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),table.data_ptr<int32_t>(),lens.data_ptr<int32_t>(),positions.data_ptr<int32_t>(),ptr,out.data_ptr<scalar_t>(),q.size(1),q.size(2),q.size(3),qr.size(3),cache.size(1),table.size(1),nk,scale,causal,indices.has_value()); });
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
