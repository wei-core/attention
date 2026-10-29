import math
import pytest
import torch
from qd_attention.mla import mla_query_absorb, mla_value_project, latent_kv_write, mla_decode
from qd_attention.reference.mla import mla_reference
from qd_attention import is_cuda_extension_loaded

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and is_cuda_extension_loaded()), reason='requires CUDA extension')

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('sparse',[False, True])
def test_paged_mla(dtype,sparse):
    torch.manual_seed(10)
    b,t,h,dn,dr,dc,dv,L,P=2,3,3,8,6,11,7,9,4
    q=torch.randn(b,t,h,dn,device='cuda',dtype=dtype); qr=torch.randn(b,t,h,dr,device='cuda',dtype=dtype)
    latent=torch.randn(b,L,dc,device='cuda',dtype=dtype); rope=torch.randn(b,L,dr,device='cuda',dtype=dtype)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype); wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)
    table=torch.tensor([[4,0,5],[1,3,2]],device='cuda',dtype=torch.int32)
    lens=torch.tensor([9,5],device='cuda',dtype=torch.int32)
    positions=torch.tensor([[0,3,8],[-1,2,4]],device='cuda',dtype=torch.int32)
    cache=torch.zeros(6,P,dc+dr,device='cuda',dtype=dtype)
    slots=torch.tensor([int(table[bb,i//P])*P+i%P for bb in range(b) for i in range(L)],device='cuda',dtype=torch.int64)
    latent_kv_write(latent.reshape(-1,dc),rope.reshape(-1,dr),cache,slots)
    indices=torch.tensor([[[0,-1,-1],[0,2,3],[1,4,8]],[[0,-1,-1],[0,2,-1],[1,3,4]]],device='cuda',dtype=torch.int32) if sparse else None
    actual=mla_decode(q,qr,cache,table,lens,positions,wk,wv,indices=indices)
    expected=mla_reference(q,qr,latent,rope,wk,wv,positions,lens,indices=indices)
    alt=mla_reference(q,qr,latent,rope,wk,wv,positions,lens,absorbed=True,indices=indices)
    torch.testing.assert_close(expected,alt,atol=2e-5,rtol=2e-5)
    tol=4e-4 if dtype==torch.float32 else .07 if dtype==torch.float16 else .4
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)


def test_mla_projection_and_contract():
    q=torch.randn(1,2,3,5,device='cuda'); wk=torch.randn(3,5,7,device='cuda'); wv=torch.randn(3,4,7,device='cuda')
    qa=mla_query_absorb(q,wk)
    torch.testing.assert_close(qa,torch.einsum('bthd,hdc->bthc',q,wk),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(mla_value_project(qa,wv),torch.einsum('bthc,hvc->bthv',qa,wv),atol=1e-5,rtol=1e-5)
    with pytest.raises(RuntimeError,match='dimension'):
        mla_query_absorb(q,torch.randn(3,6,7,device='cuda'))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_direct_prefill(dtype):
    from qd_attention.mla import mla_prefill
    torch.manual_seed(12);b,t,h,dn,dr,dc,dv,L=2,3,2,8,6,11,7,9
    q=torch.randn(b,t,h,dn,device='cuda',dtype=dtype);qr=torch.randn(b,t,h,dr,device='cuda',dtype=dtype)
    c=torch.randn(b,L,dc,device='cuda',dtype=dtype);kr=torch.randn(b,L,dr,device='cuda',dtype=dtype)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype);wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)
    lens=torch.tensor([9,5],device='cuda',dtype=torch.int32);pos=torch.tensor([[0,3,8],[-1,2,4]],device='cuda',dtype=torch.int32)
    actual=mla_prefill(q,qr,c,kr,wk,wv,lens,pos); expected=mla_reference(q,qr,c,kr,wk,wv,pos,lens)
    tol=3e-4 if dtype==torch.float32 else .02 if dtype==torch.float16 else .08
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)


def test_mla_typical_latent_width():
    torch.manual_seed(55);b,t,h,dn,dr,dc,dv,L=1,1,2,32,64,512,64,17
    q=torch.randn(b,t,h,dn,device='cuda',dtype=torch.bfloat16);qr=torch.randn(b,t,h,dr,device='cuda',dtype=torch.bfloat16)
    c=torch.randn(b,L,dc,device='cuda',dtype=torch.bfloat16);kr=torch.randn(b,L,dr,device='cuda',dtype=torch.bfloat16)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=torch.bfloat16)/math.sqrt(dn);wv=torch.randn(h,dv,dc,device='cuda',dtype=torch.bfloat16)/math.sqrt(dc)
    table=torch.tensor([[1,0]],device='cuda',dtype=torch.int32);cache=torch.zeros(2,16,dc+dr,device='cuda',dtype=torch.bfloat16)
    slots=torch.tensor(list(range(16,32))+[0],device='cuda',dtype=torch.int64);latent_kv_write(c.reshape(L,dc),kr.reshape(L,dr),cache,slots)
    lens=torch.tensor([L],device='cuda',dtype=torch.int32);positions=torch.tensor([[16]],device='cuda',dtype=torch.int32)
    got=mla_decode(q,qr,cache,table,lens,positions,wk,wv);ref=mla_reference(q,qr,c,kr,wk,wv,positions,lens)
    torch.testing.assert_close(got,ref,atol=.05,rtol=.08)

@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_mla_split_long_context(sparse,dtype):
    from qd_attention.mla import mla_latent_attention
    torch.manual_seed(317);L=544;dc=33;dr=9;h=2
    q=torch.randn(1,2,h,dc,device='cuda',dtype=dtype)
    qr=torch.randn(1,2,h,dr,device='cuda',dtype=dtype)
    cache=torch.randn(34,16,dc+dr,device='cuda',dtype=dtype)
    table=torch.arange(34,device='cuda',dtype=torch.int32).reshape(1,-1)
    lens=torch.tensor([L],device='cuda',dtype=torch.int32)
    positions=torch.tensor([[100,543]],device='cuda',dtype=torch.int32)
    indices=torch.arange(512,device='cuda',dtype=torch.int32).reshape(1,1,-1).expand(1,2,-1).contiguous() if sparse else None
    got=mla_latent_attention(q,qr,cache,table,lens,positions,scale=.1,indices=indices)
    c=cache.reshape(1,L,dc+dr); scores=(torch.einsum('bthd,bld->bthl',q.float(),c[:,:,:dc].float())+torch.einsum('bthd,bld->bthl',qr.float(),c[:,:,dc:].float()))*.1
    mask=torch.arange(L,device='cuda')[None,None,None,:]<=positions[:,:,None,None]
    if sparse:mask=mask & (torch.arange(L,device='cuda')[None,None,None,:]<512)
    ref=torch.einsum('bthl,bld->bthd',scores.masked_fill(~mask,-torch.inf).softmax(-1),c[:,:,:dc].float()).to(dtype)
    torch.testing.assert_close(got,ref,atol=2e-5 if dtype==torch.float32 else .02,rtol=2e-5 if dtype==torch.float32 else .02)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_mla_prefill_tiled_query_and_latent(dtype):
    from qd_attention.mla import mla_prefill
    torch.manual_seed(550);b,t,h,dn,dr,dc,dv,L=2,9,2,7,5,33,9,21
    q=torch.randn(b,t,h,dn,device='cuda',dtype=dtype);qr=torch.randn(b,t,h,dr,device='cuda',dtype=dtype)
    c=torch.randn(b,L,dc,device='cuda',dtype=dtype);kr=torch.randn(b,L,dr,device='cuda',dtype=dtype)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype);wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)
    lens=torch.tensor([21,3],device='cuda',dtype=torch.int32);pos=torch.tensor([[0,1,3,5,9,13,16,19,20],[-1,0,1,2,2,2,2,2,2]],device='cuda',dtype=torch.int32)
    actual=mla_prefill(q,qr,c,kr,wk,wv,lens,pos);ref=mla_reference(q,qr,c,kr,wk,wv,pos,lens)
    tol=4e-4 if dtype==torch.float32 else .03 if dtype==torch.float16 else .1
    torch.testing.assert_close(actual,ref,atol=tol,rtol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('causal',[False,True])
def test_fused_mla_decode_matches_unfused(dtype,sparse,causal):
    from qd_attention.mla import mla_latent_attention
    torch.manual_seed(79);b,h,dn,dr,dc,dv,L,P=2,3,17,9,33,15,32,16
    q=torch.randn(b,1,h,dn,device='cuda',dtype=dtype);qr=torch.randn(b,1,h,dr,device='cuda',dtype=dtype)
    cache=torch.randn(4,P,dc+dr,device='cuda',dtype=dtype);table=torch.tensor([[2,0],[3,1]],device='cuda',dtype=torch.int32)
    lens=torch.tensor([32,0],device='cuda',dtype=torch.int32);pos=torch.tensor([[17],[-1]],device='cuda',dtype=torch.int32)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype);wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)
    ix=torch.tensor([[[0,4,17,18,-1]],[[0,1,-1,-1,-1]]],device='cuda',dtype=torch.int32) if sparse else None
    scale=1/math.sqrt(dn+dr)
    ref=mla_value_project(mla_latent_attention(mla_query_absorb(q,wk),qr,cache,table,lens,pos,scale=scale,causal=causal,indices=ix),wv)
    actual=mla_decode(q,qr,cache,table,lens,pos,wk,wv,scale=scale,causal=causal,indices=ix)
    tol=1e-4 if dtype==torch.float32 else .02 if dtype==torch.float16 else .12
    torch.testing.assert_close(actual,ref,atol=tol,rtol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('causal',[False,True])
def test_fused_split_decode_matches_unfused(dtype,sparse,causal):
    from qd_attention.mla import mla_latent_attention
    torch.manual_seed(833);b,h,dn,dr,dc,dv,L,P=2,2,9,7,33,11,544,16
    q=torch.randn(b,1,h,dn,device='cuda',dtype=dtype);qr=torch.randn(b,1,h,dr,device='cuda',dtype=dtype)
    cache=torch.randn(68,P,dc+dr,device='cuda',dtype=dtype);table=torch.arange(68,device='cuda',dtype=torch.int32).reshape(b,-1)
    lens=torch.tensor([L,0],device='cuda',dtype=torch.int32);pos=torch.tensor([[499],[-1]],device='cuda',dtype=torch.int32)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype)/math.sqrt(dn);wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)/math.sqrt(dc)
    ix=torch.arange(512,device='cuda',dtype=torch.int32).reshape(1,1,-1).expand(b,1,-1).contiguous() if sparse else None
    scale=1/math.sqrt(dn+dr)
    ref=mla_value_project(mla_latent_attention(mla_query_absorb(q,wk),qr,cache,table,lens,pos,scale=scale,causal=causal,indices=ix),wv)
    got=mla_decode(q,qr,cache,table,lens,pos,wk,wv,scale=scale,causal=causal,indices=ix)
    tol=1e-4 if dtype==torch.float32 else .02 if dtype==torch.float16 else .05
    torch.testing.assert_close(got,ref,atol=tol,rtol=tol)

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('dc,dr',[(33,7),(64,16)])
def test_mla_prefill_wmma_tiles(dtype,dc,dr):
    from qd_attention.mla import mla_prefill
    torch.manual_seed(778);b,t,h,dn,dv,L=2,19,2,16,16,35
    q=torch.randn(b,t,h,dn,device='cuda',dtype=dtype);qr=torch.randn(b,t,h,dr,device='cuda',dtype=dtype)
    c=torch.randn(b,L,dc,device='cuda',dtype=dtype);kr=torch.randn(b,L,dr,device='cuda',dtype=dtype)
    wk=torch.randn(h,dn,dc,device='cuda',dtype=dtype)/math.sqrt(dn);wv=torch.randn(h,dv,dc,device='cuda',dtype=dtype)/math.sqrt(dc)
    lens=torch.tensor([L,0],device='cuda',dtype=torch.int32);pos=torch.arange(L-t,L,device='cuda',dtype=torch.int32).repeat(b,1)
    got=mla_prefill(q,qr,c,kr,wk,wv,lens,pos,backend="wmma");ref=mla_reference(q,qr,c,kr,wk,wv,pos,lens)
    tol=.025 if dtype==torch.float16 else .08
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got,ref,atol=tol,rtol=tol)


def test_mla_prefill_wmma_rejects_unsupported_dtype():
    from qd_attention.mla import mla_prefill
    q=torch.randn(1,16,1,4,device='cuda');qr=torch.randn(1,16,1,4,device='cuda');latent=torch.randn(1,16,4,device='cuda');rope=torch.randn(1,16,4,device='cuda');wk=torch.randn(1,4,4,device='cuda');wv=torch.randn(1,4,4,device='cuda');lens=torch.tensor([16],device='cuda',dtype=torch.int32);pos=torch.arange(16,device='cuda',dtype=torch.int32).reshape(1,16)
    with pytest.raises(RuntimeError,match='WMMA requires'):
        mla_prefill(q,qr,latent,rope,wk,wv,lens,pos,backend='wmma')
