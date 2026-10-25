import math
import torch
import torch.nn.functional as F
import pytest

from qd_attention.reference import reference_gqa_decode
from qd_attention.testing import make_decode_case


@pytest.mark.parametrize("lengths", [[0, 1, 7, 19], [16, 31]])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2), (8, 1)])
def test_reference_matches_sdp_cpu(lengths, q_heads, kv_heads):
    q, k, v, table, seq_lens = make_decode_case(
        lengths=lengths, q_heads=q_heads, kv_heads=kv_heads,
        head_dim=32, block_size=8, dtype=torch.float32,
    )
    got = reference_gqa_decode(q, k, v, table, seq_lens)
    expected = torch.zeros_like(q)
    for b, length in enumerate(lengths):
        if length == 0:
            continue
        pages = table[b, :math.ceil(length / 8)].long()
        kk = k.index_select(0, pages).reshape(-1, kv_heads, 32)[:length]
        vv = v.index_select(0, pages).reshape(-1, kv_heads, 32)[:length]
        kk = kk.transpose(0, 1).repeat_interleave(q_heads // kv_heads, dim=0)
        vv = vv.transpose(0, 1).repeat_interleave(q_heads // kv_heads, dim=0)
        expected[b] = F.scaled_dot_product_attention(
            q[b, :, None, :], kk, vv, is_causal=False
        ).squeeze(-2)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
