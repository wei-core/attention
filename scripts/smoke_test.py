"""End-to-end operator correctness smoke check; requires compiled CUDA extension."""

import torch
import qd_attention as qda
from qd_attention.testing import make_decode_case


def main():
    if not torch.cuda.is_available() or not qda.is_cuda_extension_loaded():
        raise SystemExit("CUDA GPU or compiled qd_attention extension is unavailable")
    tensors = make_decode_case(
        lengths=[0, 1, 17, 65], q_heads=8, kv_heads=2, head_dim=64,
        block_size=16, dtype=torch.float16, device="cuda",
    )
    expected = qda.reference_gqa_decode(*tensors)
    actual = qda.gqa_decode(*tensors)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    print("PASS: paged GQA Decode (FP16), mixed lengths incl. zero, non-contiguous physical pages")


if __name__ == "__main__":
    main()
