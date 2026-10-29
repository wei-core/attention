# RTX 5060 correctness and performance measurements

2026-10-29 (Asia/Shanghai), RTX 5060 8 GiB, compute capability 12.0, PyTorch 2.9.1+cu128, CUDA Toolkit 12.8, Triton 3.5.1, FlashAttention 2.8.3. The desktop/Xwayland was present and occupied about 2 GiB; clocks were not locked and the device was not isolated. Results are measurements of this machine and should be repeated for performance decisions.

`benchmark.csv`, `raw.json` and `environment.json` contain the BF16/FP16/FP32 × B=1/4/16 × L=128/1024/4096 baseline matrix. It used 20 warmups, five samples, and ten operator calls per sample. CUDA events cover the real Python operator path including launches and output allocation, rather than an isolated kernel. The saved Git HEAD was `2e4ec76` **with uncommitted modifications** listed in `environment.json`; it does not identify the measured sources by itself. Relevant measured implementation hashes:

```text
25ccb5954510bb842450f33543c484962c4b357428210aad0b790f322098c4b7  csrc/dense/gqa_decode.cu
936f82e488594a3808fd6ada7fa4cb945a96ab6577e8aacedc36ba6728d32580  csrc/common/online_softmax.cuh
9a30b275b352dde42ccbaf4e7f41f5016ec1a79403181498b45057242457635c  csrc/dense/gqa_prefill.cu
2765348ab90667d88ee8284fe1a14633b785b11db9627b0b051f76731eb06df4  qd_attention/triton_decode.py
c1314a42301562bc2c0d6c447c478454550361bf11fa0c9fbe84dc68b2cafa57  qd_attention/triton_prefill.py
```

`stage2_comparison.json` records five warmups and three samples of five calls for each BF16 case, using the same paged inputs for baseline and split decode. SDPA's native case uses pre-gathered contiguous K/V; the gather-only and gather-plus-SDPA paths separately include the layout conversion. The Prefill comparisons use the same dense causal inputs and check numerical results before timing. Python dispatch and any GPU idle gaps between launches can affect event intervals; these are not pure single-kernel measurements. The sample counts are exploratory and no statistical significance is claimed.

For example, B=1, Hq=32, Hkv=8, D=128, L=4096: sequential CUDA baseline median 2635.7 µs, token-parallel Triton with 8 splits and preallocated workspace 106.8 µs. B=16, same remaining dimensions: baseline 8608.1 µs, Triton 8 splits 1386.2 µs. These comparisons are particular to the recorded workloads and include different operator launch counts. The current automatic split heuristic is not proven optimal.

For BF16 causal Prefill B=1, L=512, Hq=8, Hkv=2, D=64, the recorded operator medians were 965.2 µs for the SIMT CUDA path, 64.7 µs for Triton Tensor Core, 43.9 µs for PyTorch SDPA, and 128.2 µs for FlashAttention 2.8.3. These measurements include different wrappers and launch overhead, not just math kernels.

`nsys_cuda.nsys-rep` and `nsys_triton.nsys-rep` contain separate Nsight Systems traces of BF16 B=1, L=1024; see the matching text summaries. The baseline kernel median was 672.4 µs across 7 launches. The Triton trace measured 15 `_decode` launches (median 27.7 µs) and 15 `_merge` launches (median 0.75 µs). Warmup launches are included, and these profiles were collected separately, so this is not a controlled end-to-end comparison.

The tested Triton Prefill compiler output contains `mma.sync.aligned.m16n8k16` PTX for both FP16 and BF16. PTX confirms the generated instruction path, not achieved Tensor Core utilization. Nsight Compute metrics are needed for occupancy, bandwidth, spilling and utilization claims.

Reproduce after building the extension and installing compatible Triton/FlashAttention:

```bash
python benchmarks/measure_stage1.py --output results/NEW_DIRECTORY
python benchmarks/compare_stage2.py --output results/NEW_DIRECTORY/stage2_comparison.json
nsys profile --trace=cuda,nvtx --sample=none -o /tmp/qd-triton \
  python benchmarks/bench_gqa_decode.py --batch 1 --q-heads 32 --kv-heads 8 \
  --head-dim 128 --seq-len 1024 --backend triton --splits 4
```

The full CUDA test suite on the recorded environment passed 96 tests with no skips. Metadata values on GPU are caller-owned; the optional synchronizing validator checks paged lengths, page IDs and duplicate append destinations in debug workflows.

## Nsight Compute limitation

Nsight Compute 2026.3.1 attempted one B=1/L=1024 Triton `_decode` launch with the `basic` set. Both normal execution and `sudo` returned `ERR_NVGPUCTRPERM`; see `ncu_permission.log`. No counter-based report was produced. On this WSL system the Windows host NVIDIA driver controls access to GPU performance counters; enable it on the host per [NVIDIA instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM), then rerun:

```bash
/home/ubuntu/.local/opt/qd-nsight/opt/nvidia/nsight-compute/2026.3.1/ncu \
  --target-processes all --kernel-name-base function --kernel-name regex:_decode \
  --launch-skip 1 --launch-count 1 --set basic -o results/ncu_decode \
  python benchmarks/bench_gqa_decode.py --batch 1 --q-heads 32 --kv-heads 8 \
  --head-dim 128 --seq-len 1024 --warmup 2 --repeat 1 --backend triton --splits 4
```
