import pytest
import torch
import qd_attention as qda
from qd_attention.testing import make_decode_case

CUDA_READY = torch.cuda.is_available() and qda.is_cuda_extension_loaded()


@pytest.mark.skipif(not CUDA_READY, reason="requires compiled CUDA extension and GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_paged_decode_matches_reference(dtype, head_dim):
    inputs = make_decode_case(
        lengths=[0, 1, 15, 16, 17, 49, 131],
        q_heads=8, kv_heads=2, head_dim=head_dim,
        block_size=16, dtype=dtype, device="cuda",
    )
    actual = qda.gqa_decode(*inputs)
    expected = qda.reference_gqa_decode(*inputs)
    atol, rtol = (3e-4, 3e-4) if dtype == torch.float32 else (
        (2e-2, 2e-2) if dtype == torch.float16 else (6e-2, 6e-2)
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUDA_READY, reason="requires compiled CUDA extension and GPU")
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 1), (16, 4)])
def test_head_mappings(q_heads, kv_heads):
    inputs = make_decode_case(
        lengths=[33, 64], q_heads=q_heads, kv_heads=kv_heads,
        head_dim=64, block_size=16, device="cuda", dtype=torch.float32,
    )
    actual = qda.gqa_decode(*inputs)
    expected = qda.reference_gqa_decode(*inputs)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not CUDA_READY, reason="requires compiled CUDA extension and GPU")
def test_bad_metadata_dtype_raises():
    q, k, v, pages, lengths = make_decode_case(
        lengths=[8], device="cuda", dtype=torch.float16
    )
    with pytest.raises(RuntimeError, match="int32"):
        qda.gqa_decode(q, k, v, pages.long(), lengths)

@pytest.mark.skipif(not CUDA_READY, reason="requires compiled CUDA extension and GPU")
@pytest.mark.parametrize('dtype,batch,length,block_size', [
    (torch.float32,1,127,32), (torch.float16,4,128,16),
    (torch.bfloat16,16,129,32), (torch.float32,1,1024,16),
    (torch.float16,4,1024,32), (torch.bfloat16,1,4096,16),
])
def test_extended_length_matrix(dtype,batch,length,block_size):
    inputs = make_decode_case(lengths=[length]*batch,q_heads=8,kv_heads=2,head_dim=64,block_size=block_size,dtype=dtype,device='cuda')
    actual = qda.gqa_decode(*inputs)
    expected = qda.reference_gqa_decode(*inputs)
    tol = 3e-4 if dtype == torch.float32 else 2e-2 if dtype == torch.float16 else 6e-2
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
