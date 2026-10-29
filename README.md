# QD Attention — Qwen / DeepSeek 专用 CUDA Attention 算子库

**定位：独立算子库。** 当前不依赖 nano-vLLM，不加载完整模型，不开发分布式调度系统。未来是否集成推理框架是可选项。

## v0.1.0 范围

| 模块 | 状态 | 内容 |
|---|---|---|
| Dense Attention | **已实现一个正确性基线** | `gqa_decode`：Paged KV + MHA/MQA/GQA，单 CTA/Query Head，在线 Softmax |
| MLA | 规划中 | Latent KV / Matrix Absorption / MLA Decode |
| Linear | 规划中 | Gated DeltaNet：Chunkwise、Recurrent |
| Sparse | 规划中 | DSA：Indexer、Top-K、Sparse MLA |
| Compressed | 规划中 | CSA / HCA 等新结构（以确定的官方数学定义为准） |
| Common / Cache | 基础设施 | Warp Reduce、单 Token KV Append、Split-KV；量化待实现 |

当前 `gqa_decode` **不是**高性能版本，而是用于验证完整研发链路的可编译、可测试的 CUDA 基线。它顺序遍历历史 KV，不使用 Tensor Core，主路径没有自动 Split-KV dispatch，不适合与 FlashInfer 比性能。

## 适用环境

- Linux x86_64（建议 Ubuntu 或 WSL2 Ubuntu），NVIDIA RTX 5060；后续迁移 RTX 5090。
- CUDA Toolkit **12.8 或更新**（nvcc 需要识别 `sm_120`）；NVIDIA 驱动与 Toolkit 兼容。
- CUDA 版 PyTorch（`torch.version.cuda` 不应是 `None`），其 CUDA 编译版本应与本机 Toolkit **兼容**；Python >= 3.10。
- C++17 编译器、`ninja`（建议）、`pytest`。

RTX 5060/5090 都是 `sm_120`。初期不要将 `sm_100` 的优化内核直接当作 GeForce Blackwell 可用。

## 安装（RTX 5060 开发机器）

```bash
# 在适配的 Linux CUDA + PyTorch 环境中执行
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"

# PyTorch CUDA 版需要按当前驱动、Toolkit、Python 版本单独安装
python -m pip install -U pip setuptools wheel ninja pytest
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install -v --no-build-isolation -e .
```

`--no-build-isolation` 是刻意的：让扩展使用当前环境中**已经安装且可用**的 PyTorch/CUDA，而不是在隔离构建环境里重新拉取 PyTorch。

## 验证

```bash
python scripts/check_env.py
python scripts/smoke_test.py

# 更完整的单元测试
# CPU 参考实现（不需要 nvcc；如果只是阅读源码也可执行）
python -m pytest -q tests/test_reference.py

# CUDA 实现与参考值对齐（需要正确编译扩展以及 NVIDIA GPU）
python -m pytest -q tests/test_gqa_decode_cuda.py

# 内核基线计时（CUDA Events，不包含生成数据、拷贝和 Python 参考实现）
python benchmarks/bench_gqa_decode.py --batch 4 --q-heads 16 --kv-heads 4 \
    --head-dim 128 --seq-len 1024 --block-size 16 --dtype bf16
```

## 当前 API

```python
import qd_attention as qda

# q [B, Hq, D], k/v cache [Nblocks, BlockSize, Hkv, D]
# block_table [B, MaxPages], context_lens [B]
out = qda.gqa_decode(q, k_cache, v_cache, block_table, context_lens)
```

支持 FP32/FP16/BF16，D<=256，Hq 是 Hkv 的整数倍；数据连续，`block_table` / `context_lens` 使用 GPU 上的 `torch.int32`。输入和输出都是 CUDA Tensor。

当前假设 Q 和 KV 已完成相应的位置编码；本算子不包含 Q/K/V projection、RoPE、Mask 生成或 KV 写入。因其为最后一步单 Token Decode，全部有效 KV 位置都可被 Query 访问，不需要额外因果 Mask。

## 目录与扩展位置

查看 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 和 [docs/KERNEL_CONTRACT.md](docs/KERNEL_CONTRACT.md)。

## 质量边界

- 已提供 CPU 参考测试；已在 RTX 5060 上编译并运行 CUDA 正确性测试；测试覆盖不代表所有输入均已验证。
- 当前仅有 CUDA Forward，无 Backward、Torch Compile/FakeTensor、FP8/FP4、CUDA Graph 适配承诺。
- 后续每个新 Kernel 先编写 Reference + Contract + Tests，再加入 CUDA 实现和 Benchmark。

## 实验性阶段二进展

`qda.gqa_decode_split(..., splits=4, backend="triton")` 提供按 Token 并行的分页 Decode，可传入可复用的 FP32 workspace。`qda.gqa_decode_auto(...)` 使用基于 KV 容量、头数和 SM 数量的初步调度策略；该策略尚未覆盖所有工作负载的调优。`qda.kv_append(...)` 原地写入单 Token KV。输入和调用方责任见 [算子契约](docs/KERNEL_CONTRACT.md)。

`qda.gqa_prefill(...)` 提供 CUDA 基础路径，`backend="tensorcore"` 提供可选 Triton FP16/BF16 分块路径。后者需要安装 Triton；对 CUDA 长度输入采用右下对齐因果语义，详见 [Prefill 契约](docs/PREFILL_CONTRACT.md)。在 RTX 5060 的测试中生成的 PTX 包含 FP16/BF16 `mma.sync` 指令。当前均为 forward 算子，不含 Nano-vLLM 集成。性能结果和限制见 [测量记录](results/stage1_20261029/README.md)。

各公开算子的清单、性能对照与优化思路见 [Attention 算子与性能报告](docs/ATTENTION_OPERATORS_AND_BENCHMARKS.md)。
