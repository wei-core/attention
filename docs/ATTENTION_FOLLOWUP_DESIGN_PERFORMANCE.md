# Delta 与 MLA 后续优化：设计、源码及性能对照

本次以 Git 提交 `ed8fbdc` 为旧版，当前文件为新版。旧版源码在本报告中提供完整快照，新版链接指向仓库实际源码；旧快照逐字节来自该提交，且保留 [SHA-256](source_history/ed8fbdc/SHA256SUMS)。此前从 `de89326` 到 `ed8fbdc` 的修改和性能见 [上一轮报告](ATTENTION_OPTIMIZATION_20261029.md)。

## 原始文件与修改后文件

| 文件 | 原始文件（ed8fbdc） | 修改后文件 | 修改思路 |
| --- | --- | --- | --- |
| Delta Chunkwise | [完整原始文件](source_history/ed8fbdc/csrc/qwen/delta_chunkwise.cu.txt) | [修改后文件](../csrc/qwen/delta_chunkwise.cu) | 按 8 行解对角块，然后并行更新剩余行的多个右端列；矩阵组织仍为 FP32 SIMT。 |
| MLA Prefill | [完整原始文件](source_history/ed8fbdc/csrc/mla/mla_prefill.cu.txt) | [修改后文件](../csrc/mla/mla_prefill.cu) | 每次将 4 个 latent/rope token 装入共享内存，让 4 个 Query Warp 复用，减少同步次数；共享内存不满足条件时回退旧路径。 |
| MLA Decode | [完整原始文件](source_history/ed8fbdc/csrc/mla/mla.cu.txt) | [修改后文件](../csrc/mla/mla.cu) | 短上下文的 Query Absorption、分页 Attention 和 Value Projection 在一个 CTA 内完成；中间值按原语义保留输入 dtype 的舍入。 |
| PyTorch 注册 | [完整原始文件](source_history/ed8fbdc/csrc/binding/torch_ops.cpp.txt) | [修改后文件](../csrc/binding/torch_ops.cpp) | 注册融合算子，保留原有公共算子。 |
| Python 调度 | [完整原始文件](source_history/ed8fbdc/qd_attention/mla.py.txt) | [修改后文件](../qd_attention/mla.py) | 单 Query 的短 dense 或少量 sparse 候选使用融合路径；长 dense 及大 sparse 候选仍走原有分段路径。 |

原始文件扩展名附加 `.txt`，其内容不做转换。可用 `git diff ed8fbdc HEAD -- csrc/qwen/delta_chunkwise.cu csrc/mla/mla_prefill.cu csrc/mla/mla.cu csrc/binding/torch_ops.cpp qd_attention/mla.py` 查看精确差异。

## 算法细节

Delta 的块内每个 RHS 列仍按依赖顺序求解；块间通过并行矩阵式更新计算后续行。该实现不是 MMA/WMMA，仍需进一步实现和验证 Tensor Core 求解，不能把当前改动称为矩阵指令路径。

MLA Prefill 保留 Query Absorption 和 Value 后置投影：4 个 Query Warp 按相同的 latent token 块计算各自在线 Softmax，边界、不同位置和空行各自掩码。当前 QK/PV 仍是 Warp 归约与 SIMT 加权，并非 MMA 矩阵乘。

MLA Decode 对每个 Query/Head 创建一个 CTA，在线计算 Absorption、访问 paged cache 和稀疏索引、累积 latent 并投影输出。为匹配旧接口，吸收后的 Query 与归一化 latent 输出在各自原有精度转换点舍入。多 Query、长 dense 和超过 511 个 sparse 索引使用原实现；未实现在分段模式下融合 Query 和 Value 投影。

## 性能测量

RTX 5060 8 GiB，PyTorch/CUDA 环境一致；旧版为 `ed8fbdc` 的源码及构建扩展。运行顺序旧→新→新→旧，每个独立进程预热 5 次、7 组 CUDA Event，每组 10 次调用，表中取各进程中位数的中位数。算子耗时含输出/工作区分配和可能的主机提交间隙；GPU 未隔离且未锁频，少量差异无法排除波动。原始数据：[旧 1](../results/optimization_followup_20261029/before1.json)、[旧 2](../results/optimization_followup_20261029/before2.json)、[新 1](../results/optimization_followup_20261029/after1.json)、[新 2](../results/optimization_followup_20261029/after2.json)。

| 算子 | 形状（未标注为 FP32） | 原始 μs | 修改后 μs | 延迟变化 |
| --- | --- | ---: | ---: | ---: |
| gqa_prefill_cuda | L=128, H=8, Hkv=2, D=64, dtype=bf16 | 83.19 | 82.04 | -1.4% |
| gqa_prefill_triton | L=128, H=8, Hkv=2, D=64, dtype=bf16 | 59.61 | 61.86 | +3.8% |
| gqa_prefill_cuda | L=512, H=8, Hkv=2, D=64, dtype=bf16 | 618.91 | 630.31 | +1.8% |
| gqa_prefill_triton | L=512, H=8, Hkv=2, D=64, dtype=bf16 | 98.45 | 61.56 | -37.5% |
| gqa_prefill_cuda | L=1024, H=8, Hkv=2, D=64, dtype=bf16 | 1928.21 | 1987.52 | +3.1% |
| gqa_prefill_triton | L=1024, H=8, Hkv=2, D=64, dtype=bf16 | 115.77 | 115.76 | -0.0% |
| delta_chunkwise | B=2, T=32, H=4, D=16, chunk=16 | 57.26 | 53.80 | -6.0% |
| delta_chunkwise | B=2, T=32, H=4, D=64, chunk=16 | 259.54 | 263.05 | +1.4% |
| mla_prefill | B=1, T=4, H=4, L=64, Dn=16, Dr=8, Dc=32, Dv=16 | 45.94 | 44.58 | -3.0% |
| mla_decode | B=1, T=1, H=4, L=64, Dc=32, Dr=8 | 100.65 | 55.67 | -44.7% |
| indexer | B=1, T=1, L=64, H=4, D=16 | 19.01 | 19.12 | +0.6% |
| topk | B=1, T=1, L=64, K=16 | 21.01 | 20.50 | -2.4% |
| sparse_mla | B=1, T=1, L=64, K=16 | 59.32 | 24.08 | -59.4% |
| dsa_complete | B=1, T=1, L=64, K=16 | 93.99 | 64.16 | -31.7% |
| mla_prefill | B=1, T=4, H=4, L=1024, Dn=16, Dr=8, Dc=32, Dv=16 | 621.15 | 617.28 | -0.6% |
| mla_decode | B=1, T=1, H=4, L=1024, Dc=32, Dr=8 | 210.82 | 219.71 | +4.2% |
| indexer | B=1, T=1, L=1024, H=4, D=16 | 19.21 | 18.30 | -4.8% |
| topk | B=1, T=1, L=1024, K=16 | 39.71 | 39.21 | -1.2% |
| sparse_mla | B=1, T=1, L=1024, K=16 | 62.54 | 25.55 | -59.1% |
| dsa_complete | B=1, T=1, L=1024, K=16 | 209.54 | 75.00 | -64.2% |
| mla_prefill_tiled | B=1, T=16, H=4, L=256, Dn=16, Dr=8, Dc=32, Dv=16 | 194.80 | 151.30 | -22.3% |
| gated_attention | B=2, T=256, H=4, D=64 | 28.04 | 19.77 | -29.5% |
| delta_output | B=2, T=256, H=4, D=64 | 21.78 | 20.73 | -4.8% |
| causal_conv1d | B=2, T=256, C=128, K=4 | 21.04 | 20.06 | -4.6% |
| kv_append | B=4, H=8, D=128, dtype=bf16 | 17.01 | 13.74 | -19.2% |

变化为 `修改后/原始−1`。未修改的算子也列出，用于观察环境波动；短操作和长上下文未改变路径上的差异不应归因于本次改动。`dsa_complete` 包含 Indexer、Top-K 与 Sparse MLA 的完整链路。

## 正确性和剩余验证

本机完整回归 **189 passed、0 skipped**。新增测试比较融合与原分步 MLA Decode，覆盖 FP32/FP16/BF16、dense/sparse、causal/noncausal、无有效 token 和索引越界掩码；已有测试覆盖 Delta 跨 chunk、各种 dtype、不同状态和非整 Warp 维度，以及 Prefill 多 Query 的变长和掩码。没有完整模型权重、在线服务或反向传播验证。

GPU 硬件计数器仍不可用，既有 [权限错误记录](../results/optimization_20261029/ncu_attempt.log) 返回 `ERR_NVGPUCTRPERM`；不能提供真实 DRAM Roofline 或 Tensor Core 利用率。现有的 [逻辑模型](ATTENTION_METRICS_ROOFLINE.md) 是估算，不能将它误写为实测效率。启用宿主机性能计数器后，应对各 kernel 的真实执行 FLOP、DRAM 字节、活跃 warp、缓存和同步停顿进行采样，再判断瓶颈。

复现：独立导出并编译 `ed8fbdc`，然后分别执行 `python benchmarks/bench_optimization.py --root /path/to/ed8fbdc --output before.json` 和 `python benchmarks/bench_optimization.py --root "$PWD" --output after.json`；正确性运行 `python -m pytest -q`。数据、模型、状态和权重为脚本内固定种子的合成输入。
