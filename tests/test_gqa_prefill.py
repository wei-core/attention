import pytest
import torch
import qd_attention as qda
from qd_attention.reference.gqa import reference_gqa_prefill

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and qda.is_cuda_extension_loaded()), reason="requires CUDA extension")

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('heads', [(4,4), (4,1), (8,2)])
def test_prefill_matches_reference(dtype, causal, heads):
    torch.manual_seed(31)
    hq, hk = heads
    q = torch.randn(3, 7, hq, 64, device='cuda', dtype=dtype)
    k = torch.randn(3, 11, hk, 64, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    qlens = torch.tensor([7, 3, 5], dtype=torch.int32, device='cuda')
    klens = torch.tensor([11, 1, 0], dtype=torch.int32, device='cuda')
    got = qda.gqa_prefill(q, k, v, qlens, klens, causal=causal)
    expected = reference_gqa_prefill(q, k, v, qlens, klens, causal=causal)
    tol = 3e-4 if dtype == torch.float32 else 2e-2 if dtype == torch.float16 else 6e-2
    torch.testing.assert_close(got, expected, atol=tol, rtol=tol)

@pytest.mark.parametrize('d', [1, 33, 128, 256])
def test_prefill_head_dim_and_tile_boundary(d):
    q = torch.randn(1, 3, 2, d, device='cuda')
    k = torch.randn(1, 35, 1, d, device='cuda')
    v = torch.randn_like(k)
    ql = torch.tensor([3], device='cuda', dtype=torch.int32)
    kl = torch.tensor([35], device='cuda', dtype=torch.int32)
    torch.testing.assert_close(qda.gqa_prefill(q, k, v, ql, kl), reference_gqa_prefill(q,k,v,ql,kl), atol=3e-4, rtol=3e-4)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('d', [33, 64, 128])
def test_tensorcore_prefill(dtype, causal, d):
    torch.manual_seed(42)
    q = torch.randn(3, 17, 4, d, device='cuda', dtype=dtype)
    k = torch.randn(3, 35, 2, d, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    ql = torch.tensor([17, 3, 8], device='cuda', dtype=torch.int32)
    kl = torch.tensor([35, 1, 0], device='cuda', dtype=torch.int32)
    actual = qda.gqa_prefill(q, k, v, ql, kl, causal=causal, backend='tensorcore')
    expected = reference_gqa_prefill(q,k,v,ql,kl,causal=causal)
    tol = 0.015 if dtype == torch.float16 else 0.04
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

@pytest.mark.parametrize('backend,dtype', [('cuda',torch.float32),('tensorcore',torch.bfloat16)])
def test_empty_query_and_large_logits(backend,dtype):
    torch.manual_seed(7)
    q=torch.randn(2,5,4,256,device='cuda',dtype=dtype)*8
    k=torch.randn(2,37,1,256,device='cuda',dtype=dtype)*8
    v=torch.randn_like(k)
    ql=torch.tensor([0,5],device='cuda',dtype=torch.int32)
    kl=torch.tensor([37,37],device='cuda',dtype=torch.int32)
    actual=qda.gqa_prefill(q,k,v,ql,kl,backend=backend)
    expected=reference_gqa_prefill(q,k,v,ql,kl)
    assert torch.isfinite(actual).all()
    tol=3e-4 if dtype==torch.float32 else 0.06
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
