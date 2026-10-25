"""Microbenchmark: CUDA kernel wall time, not end-to-end model performance."""

import argparse
from pathlib import Path
import sys

# Supports execution from a checkout after an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import qd_attention as qda
from qd_attention.testing import make_decode_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not qda.is_cuda_extension_loaded():
        parser.error("CUDA GPU and compiled qd_attention extension are required")
    if args.seq_len < 0 or args.batch < 1 or args.repeat < 1:
        parser.error("seq_len >= 0, batch >= 1, repeat >= 1 required")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tensors = make_decode_case(
        lengths=[args.seq_len] * args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        dtype=dtype,
        device="cuda",
    )
    for _ in range(args.warmup):
        qda.gqa_decode(*tensors)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        qda.gqa_decode(*tensors)
    end.record()
    end.synchronize()
    ms_per_launch = start.elapsed_time(end) / args.repeat
    print(f"device={torch.cuda.get_device_name()}  dtype={args.dtype}")
    print(f"shape: B={args.batch} Hq={args.q_heads} Hkv={args.kv_heads} "
          f"D={args.head_dim} L={args.seq_len} Page={args.block_size}")
    print(f"gqa_decode (baseline): {ms_per_launch * 1000:.2f} us / call")
    print("Note: sequential-KV correctness baseline; not a tuned FlashDecoding kernel.")


if __name__ == "__main__":
    main()
