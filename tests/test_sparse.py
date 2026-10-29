import pytest
import torch
from qd_attention import is_cuda_extension_loaded
from qd_attention.sparse import lightning_indexer,topk_select
from qd_attention.reference.sparse import lightning_indexer_reference,topk_reference
pytestmark=pytest.mark.skipif(not(torch.cuda.is_available() and is_cuda_extension_loaded()),reason='requires CUDA extension')

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_indexer_and_topk(dtype):
    torch.manual_seed(31); q=torch.randn(2,3,4,6,device='cuda',dtype=dtype);k=torch.randn(2,9,6,device='cuda',dtype=dtype);w=torch.randn(2,3,4,device='cuda',dtype=dtype)
    lengths=torch.tensor([9,3],device='cuda',dtype=torch.int32);pos=torch.tensor([[0,4,8],[-1,2,7]],device='cuda',dtype=torch.int32)
    actual=lightning_indexer(q,k,w,lengths,pos); expected=lightning_indexer_reference(q,k,w,lengths,pos)
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(topk_select(actual,12),topk_reference(expected,12))


def test_ties_empty_and_nans():
    scores=torch.tensor([[[2.,2.,1.,float('-inf'),float('nan')], [float('-inf')]*5]],device='cuda')
    got=topk_select(scores,7)
    assert got.tolist()==[[[0,1,2,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1]]]


def test_indexer_selects_logical_indices_for_paged_mla():
    from qd_attention.sparse import sparse_mla
    from qd_attention.mla import latent_kv_write
    from qd_attention.reference.mla import mla_reference
    torch.manual_seed(43);b,t,h,dn,dr,dc,dv,l,p=1,2,2,5,4,6,3,7,4
    q=torch.randn(b,t,h,dn,device='cuda');qr=torch.randn(b,t,h,dr,device='cuda');c=torch.randn(b,l,dc,device='cuda');kr=torch.randn(b,l,dr,device='cuda')
    wk=torch.randn(h,dn,dc,device='cuda');wv=torch.randn(h,dv,dc,device='cuda');table=torch.tensor([[1,0]],device='cuda',dtype=torch.int32);cache=torch.zeros(2,p,dc+dr,device='cuda')
    slots=torch.tensor([4,5,6,7,0,1,2],device='cuda',dtype=torch.int64);latent_kv_write(c.reshape(l,dc),kr.reshape(l,dr),cache,slots)
    iq=torch.randn(b,t,3,4,device='cuda');ik=torch.randn(b,l,4,device='cuda');weights=torch.randn(b,t,3,device='cuda');lengths=torch.tensor([7],device='cuda',dtype=torch.int32);pos=torch.tensor([[2,6]],device='cuda',dtype=torch.int32)
    ix=topk_select(lightning_indexer(iq,ik,weights,lengths,pos),4)
    got=sparse_mla(q,qr,cache,table,lengths,pos,wk,wv,ix)
    expected=mla_reference(q,qr,c,kr,wk,wv,pos,lengths,indices=ix)
    torch.testing.assert_close(got,expected,atol=3e-4,rtol=3e-4)


def test_topk_nan_does_not_displace_valid_entries():
    scores=torch.tensor([[[float('nan'), 3., 2., float('inf')]]],device='cuda')
    torch.testing.assert_close(topk_select(scores,2),topk_reference(scores,2))

@pytest.mark.parametrize('features', [1, 33, 97])
def test_indexer_warp_feature_boundaries(features):
    torch.manual_seed(73)
    q = torch.randn(1, 3, 5, features, device='cuda')
    k = torch.randn(1, 67, features, device='cuda')
    weights = torch.randn(1, 3, 5, device='cuda')
    lengths = torch.tensor([65], device='cuda', dtype=torch.int32)
    positions = torch.tensor([[0, 34, 66]], device='cuda', dtype=torch.int32)
    torch.testing.assert_close(lightning_indexer(q, k, weights, lengths, positions),
                               lightning_indexer_reference(q, k, weights, lengths, positions),
                               atol=2e-4, rtol=2e-4)


def test_parallel_topk_cross_block_and_repeated_scores():
    torch.manual_seed(74)
    scores = torch.randint(-3, 4, (2, 3, 513), device='cuda').float()
    scores[0, 0, 257] = float('nan')
    torch.testing.assert_close(topk_select(scores, 520), topk_reference(scores, 520))

@pytest.mark.parametrize('k', [1, 16, 64])
def test_topk_tiled_merge(k):
    torch.manual_seed(1234)
    scores=torch.randint(-8,9,(2,3,777),device='cuda').float()
    scores[0,0,:256]=float('nan')
    torch.testing.assert_close(topk_select(scores,k),topk_reference(scores,k))
