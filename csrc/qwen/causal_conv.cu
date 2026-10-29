#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>
namespace {
template<typename T> __global__ void conv(const T* x,const T* weight,const T* initial,T* y,T* next,int tlen,int channels,int width) {
 int b=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=channels)return;
 for(int t=0;t<tlen;t++) {
  float sum=0; for(int i=0;i<width;i++) {int p=t-width+1+i;float a=p<0?float(initial[(static_cast<int64_t>(b)*channels+c)*width+width+p]):float(x[(static_cast<int64_t>(b)*tlen+p)*channels+c]);sum+=a*float(weight[c*width+i]);}
  y[(static_cast<int64_t>(b)*tlen+t)*channels+c]=static_cast<T>(sum/(1.f+expf(-sum)));
 }
 for(int i=0;i<width;i++) {int p=tlen-width+i;next[(static_cast<int64_t>(b)*channels+c)*width+i]=p<0?initial[(static_cast<int64_t>(b)*channels+c)*width+width+p]:x[(static_cast<int64_t>(b)*tlen+p)*channels+c];}
}
}
std::tuple<torch::Tensor,torch::Tensor> causal_conv1d_cuda(const torch::Tensor& x,const torch::Tensor& weight,const torch::Tensor& state) {
 for(const auto& z:{x,weight,state})TORCH_CHECK(z.is_cuda() && z.is_contiguous() && z.device()==x.device(),"inputs must be contiguous on one CUDA device");
 TORCH_CHECK(x.dim()==3 && weight.dim()==2 && state.dim()==3 && x.size(2)==weight.size(0) && state.size(0)==x.size(0) && state.size(1)==x.size(2) && state.size(2)==weight.size(1) && weight.size(1)>0 && weight.size(0)>0,"invalid input/state/weight shape");
 TORCH_CHECK(x.scalar_type()==weight.scalar_type() && x.scalar_type()==state.scalar_type() && (x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16),"dtype mismatch or unsupported dtype");
 auto out=at::empty_like(x), next=at::empty_like(state);if(!x.size(0)) return {out,next};c10::cuda::CUDAGuard guard(x.device());dim3 grid((x.size(2)+127)/128,x.size(0));
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"causal_conv",[&]{conv<scalar_t><<<grid,128,0,at::cuda::getCurrentCUDAStream().stream()>>>(x.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),state.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),next.data_ptr<scalar_t>(),x.size(1),x.size(2),weight.size(1));});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,next};
}
