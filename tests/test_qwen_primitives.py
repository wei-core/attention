import pytest
import torch
from qd_attention import is_cuda_extension_loaded
from qd_attention.qwen import gated_attention,delta_output,delta_recurrent
from qd_attention.reference.delta import gated_attention_reference,delta_output_reference,delta_recurrent_reference

pytestmark=pytest.mark.skipif(not(torch.cuda.is_available() and is_cuda_extension_loaded()),reason='requires compiled CUDA extension')

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_gate_and_output(dtype):
    torch.manual_seed(7); a=torch.randn(2,3,4,65,device='cuda',dtype=dtype); z=torch.randn_like(a); w=torch.randn(65,device='cuda',dtype=dtype)
    tol=1e-5 if dtype==torch.float32 else .02
    torch.testing.assert_close(gated_attention(a,z),gated_attention_reference(a,z),atol=tol,rtol=tol)
    torch.testing.assert_close(delta_output(a,z,w),delta_output_reference(a,z,w),atol=tol,rtol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_delta_recurrent(dtype):
    torch.manual_seed(8); q=torch.randn(2,3,17,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn(2,3,11,device='cuda',dtype=dtype)
    state=torch.randn(2,3,17,11,device='cuda'); saved=state.clone();g=-torch.rand(2,3,device='cuda');beta=torch.rand(2,3,device='cuda')
    actual,new=delta_recurrent(q,k,v,g,beta,state)
    expected,refstate=delta_recurrent_reference(q,k,v,g,beta,state)
    tol=3e-5 if dtype==torch.float32 else .02
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol);torch.testing.assert_close(new,refstate,atol=2e-5,rtol=2e-5);torch.testing.assert_close(state,saved)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('length',[0,1,3,8])
def test_causal_convolution(dtype,length):
    from qd_attention.qwen import causal_conv1d
    from qd_attention.reference.delta import causal_conv1d_reference
    torch.manual_seed(9); x=torch.randn(2,length,7,device='cuda',dtype=dtype);w=torch.randn(7,4,device='cuda',dtype=dtype);state=torch.randn(2,7,4,device='cuda',dtype=dtype)
    y,next=causal_conv1d(x,w,state); expected,refstate=causal_conv1d_reference(x,w,state)
    tol=1e-5 if dtype==torch.float32 else .03
    torch.testing.assert_close(y,expected,atol=tol,rtol=tol);torch.testing.assert_close(next,refstate)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('chunk',[16,32,64])
def test_chunkwise_matches_recurrence(dtype,chunk):
    from qd_attention.qwen import delta_chunkwise
    from qd_attention.reference.delta import delta_sequence_reference
    torch.manual_seed(41);b,t,h,dk,dv=2,19,2,5,4
    q=torch.randn(b,t,h,dk,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn(b,t,h,dv,device='cuda',dtype=dtype)
    g=-torch.rand(b,t,h,device='cuda')*.2;beta=torch.rand(b,t,h,device='cuda');state=torch.randn(b,h,dk,dv,device='cuda')
    expected,refstate=delta_sequence_reference(q,k,v,g,beta,state)
    actual,new=delta_chunkwise(q,k,v,g,beta,state,chunk_size=chunk)
    tol=2e-5 if dtype==torch.float32 else .02
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol);torch.testing.assert_close(new,refstate,atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize('length',[0,1,33,65])
def test_chunkwise_boundary_and_state_carry(length):
    from qd_attention.qwen import delta_chunkwise, delta_recurrent
    torch.manual_seed(18);b,h,dk,dv=1,2,7,6
    q=torch.randn(b,length,h,dk,device='cuda');k=torch.randn_like(q);v=torch.randn(b,length,h,dv,device='cuda')
    g=-torch.rand(b,length,h,device='cuda')*.1;beta=torch.rand(b,length,h,device='cuda');state=torch.randn(b,h,dk,dv,device='cuda')
    out,final=delta_chunkwise(q,k,v,g,beta,state,chunk_size=16)
    seq=[];s=state
    for t in range(length):
        y,s=delta_recurrent(q[:,t].contiguous(),k[:,t].contiguous(),v[:,t].contiguous(),g[:,t].contiguous(),beta[:,t].contiguous(),s)
        seq.append(y)
    expected=torch.stack(seq,1) if seq else v.new_empty(v.shape)
    torch.testing.assert_close(out,expected,atol=3e-5,rtol=3e-5);torch.testing.assert_close(final,s,atol=3e-5,rtol=3e-5)


def test_causal_conv_decode_matches_prefill():
    from qd_attention.qwen import causal_conv1d,causal_conv1d_update
    torch.manual_seed(1);x=torch.randn(2,9,5,device='cuda');w=torch.randn(5,4,device='cuda');state=torch.randn(2,5,4,device='cuda')
    expected, final=causal_conv1d(x,w,state); ys=[];s=state
    for t in range(x.size(1)):
        y,s=causal_conv1d_update(x[:,t].contiguous(),w,s);ys.append(y)
    torch.testing.assert_close(torch.stack(ys,1),expected);torch.testing.assert_close(s,final)


def test_delta_zero_key_and_zero_beta():
    q=torch.zeros(1,1,5,device='cuda');k=torch.zeros_like(q);v=torch.randn(1,1,3,device='cuda')
    state=torch.randn(1,1,5,3,device='cuda');g=torch.tensor([[-2.]],device='cuda');beta=torch.zeros(1,1,device='cuda')
    out,next=delta_recurrent(q,k,v,g,beta,state)
    assert torch.isfinite(out).all() and torch.isfinite(next).all()
    torch.testing.assert_close(out,torch.zeros_like(out));torch.testing.assert_close(next,state*g.exp()[...,None,None])
