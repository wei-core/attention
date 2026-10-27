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
