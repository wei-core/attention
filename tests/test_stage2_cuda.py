import pytest
import torch
import qd_attention as qda
from qd_attention.testing import make_decode_case

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and qda.is_cuda_extension_loaded()), reason="requires CUDA extension")

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 2, 4, 16])
def test_split_decode(dtype, splits):
    args = make_decode_case(lengths=[0, 1, 17, 65, 131], q_heads=8, kv_heads=2,
                            head_dim=64, block_size=16, dtype=dtype, device="cuda")
    result = qda.gqa_decode_split(*args, splits=splits)
    reference = qda.reference_gqa_decode(*args)
    tolerance = 3e-4 if dtype == torch.float32 else 2e-2 if dtype == torch.float16 else 6e-2
    torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_append_then_decode(dtype):
    q, k, v, table, lengths = make_decode_case(lengths=[15, 16, 33], q_heads=8, kv_heads=2,
                                                head_dim=64, block_size=16, dtype=dtype, device="cuda")
    # Use slots already allocated, including a page boundary, with unique destinations.
    positions = torch.tensor([0, 15, 16], device="cuda", dtype=torch.int32)
    key = torch.randn((3, 2, 64), device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    qda.kv_append(key, value, k, v, table, positions)
    for b, t in enumerate(positions.tolist()):
        page = table[b, t // 16].item()
        torch.testing.assert_close(k[page, t % 16], key[b])
        torch.testing.assert_close(v[page, t % 16], value[b])
    torch.testing.assert_close(qda.gqa_decode(*[q, k, v, table, lengths]),
                               qda.reference_gqa_decode(q, k, v, table, lengths),
                               atol=0.06 if dtype == torch.bfloat16 else 0.02 if dtype == torch.float16 else 3e-4,
                               rtol=0.06 if dtype == torch.bfloat16 else 0.02 if dtype == torch.float16 else 3e-4)

def test_split_reusable_workspace():
    args = make_decode_case(lengths=[0, 7, 129], q_heads=4, kv_heads=1,
                            head_dim=33, block_size=16, dtype=torch.float32, device='cuda')
    scratch = torch.empty((3, 4, 8, 35), device='cuda', dtype=torch.float32)
    for _ in range(2):
        got = qda.gqa_decode_split(*args, splits=8, workspace=scratch)
        torch.testing.assert_close(got, qda.reference_gqa_decode(*args), atol=3e-4, rtol=3e-4)
    with pytest.raises(RuntimeError, match='workspace shape'):
        qda.gqa_decode_split(*args, splits=4, workspace=scratch)

@pytest.mark.parametrize('dtype', [torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('splits', [1,2,4,16])
def test_triton_split_decode(dtype,splits):
    args = make_decode_case(lengths=[0,1,17,65,131],q_heads=8,kv_heads=2,head_dim=64,block_size=16,dtype=dtype,device='cuda')
    workspace = torch.empty((5,8,splits,66),device='cuda',dtype=torch.float32)
    result = qda.gqa_decode_split(*args,splits=splits,workspace=workspace,backend='triton')
    tol = 3e-4 if dtype == torch.float32 else 2e-2 if dtype == torch.float16 else 6e-2
    torch.testing.assert_close(result,qda.reference_gqa_decode(*args),atol=tol,rtol=tol)

@pytest.mark.parametrize('length', [17, 513, 1024])
def test_auto_dispatch(length):
    args = make_decode_case(lengths=[length],q_heads=8,kv_heads=2,head_dim=64,block_size=16,dtype=torch.float32,device='cuda')
    torch.testing.assert_close(qda.gqa_decode_auto(*args),qda.reference_gqa_decode(*args),atol=3e-4,rtol=3e-4)


def test_append_unaligned_contiguous_fallback():
    base=torch.randn(1+2*33,device='cuda');key=base[1:].view(1,2,33)
    value=torch.randn_like(key)
    cache=torch.zeros(1,4,2,33,device='cuda');other=torch.zeros_like(cache)
    table=torch.zeros(1,1,device='cuda',dtype=torch.int32);pos=torch.tensor([2],device='cuda',dtype=torch.int32)
    qda.kv_append(key,value,cache,other,table,pos)
    torch.testing.assert_close(cache[0,2],key[0]);torch.testing.assert_close(other[0,2],value[0])
