# 全部 Attention 与配套算子：性能清单与对照

本报告记录提交 `00961244a23d57b20c50b5b47fd92d3e4af080` 的 forward 算子。**本轮只整理、验证和计时，没有对算子进行优化或调参。** 测试时间为 2026-10-29（北京时间）。阶段一、二的历史结果保留在 [此前报告](ATTENTION_OPERATORS_AND_BENCHMARKS.md)；本报告覆盖后续 MLA、Qwen Hybrid 与 DSA 算子。

## 1. 先列出当前实现的算子

| 类别 | 接口 / 路径 | 计算作用 |
| --- | --- | --- |
| Dense Attention | `gqa_decode` | 分页单 Token Decode 顺序 CUDA 基线；支持 MHA/MQA/GQA |
| Dense Attention | `gqa_decode_split` CUDA / Triton | 将分页 KV 分段计算并合并在线 Softmax 状态 |
| 调度 | `gqa_decode_auto` | 根据静态页容量等信息选择 Decode 路径，不是新的数学算子 |
| Dense Attention | `gqa_prefill` CUDA / Tensor Core | Dense GQA Prefill，分别为 SIMT 和 Triton MMA 路径 |
| KV 配套 | `kv_append` | 单 Token 分页 K/V 写入；不是 Attention 数学本身 |
| MLA | `latent_kv_write` | 按物理槽写入已归一化 latent 与已旋转位置 Key |
| MLA | `mla_query_absorb`、`mla_value_project` | Query 矩阵吸收与 latent 输出 Value 投影 |
| MLA | `mla_latent_attention` | 从分页 latent 计算 Softmax 与 latent 加权结果；可用稀疏逻辑索引 |
| MLA | `mla_prefill`、`mla_decode` | 连续 Direct causal Prefill；Paged Absorbed Decode |
| Qwen Hybrid | `gated_attention`、`delta_output` | Attention 的 sigmoid gate；Delta 输出 RMSNorm + SiLU gate |
| Qwen Hybrid | `causal_conv1d`、`causal_conv1d_update` | 带卷积状态的多 Token 与单 Token depthwise causal conv + SiLU |
| Qwen Hybrid | `delta_recurrent`、`delta_chunkwise` | 单步 Delta 状态递归；分块下三角求解与状态传递 |
| DSA | `lightning_indexer`、`topk_select` | 位置评分、合法 Token 的确定性 Top-K |
| DSA | `sparse_mla` | 用逻辑索引读取分页 latent 并复用 MLA 计算 |

参考实现、Softmax/归约辅助函数不是独立公开推理算子。MHA/MQA/GQA 是 Head 配置，不是三套不同 API。Compressed、Backward、FP8、模型投影/RoPE 和 Nano-vLLM 集成不在当前实现内。接口与数据安全边界详见 [阶段三至五契约](STAGE345_CONTRACT.md)、[Dense 契约](KERNEL_CONTRACT.md)、[Prefill 契约](PREFILL_CONTRACT.md)。

## 2. 测试条件和可比性

- RTX 5060 8 GiB，SM120，Windows 驱动 617.14，WSL；桌面 Xwayland 存在，GPU 未隔离，时钟未锁定。PyTorch 2.9.1+cu128、Triton 3.5.1、FlashAttention 2.8.3。
- 每个测试均在计时前对照相应参考结果。Dense 采用 BF16，其他阶段核心采用 FP32，不能跨 dtype 直接比较。测试只覆盖下表固定形状，不代表所有输入。两次独立进程固定顺序运行；Dense 每次每路径预热 10 次，测 5 组、每组 10 次；阶段三至五预热 5 次，测 5 组、每组 5 次。
- 表内为两次运行各自中位数的中位数（两个值的均值），单位 **µs/调用**。用 CUDA Events 测量完整算子区间，包含 GPU kernel、分配和可能的 Python/主机提交空隙，并非纯 kernel 时间；固定顺序、短采样及未锁频意味着小差异不代表统计显著。
- 表中的“变化”统一为 `(算子延迟 / 本行对照延迟 - 1) × 100%`；负值表示本次测得更短。它只是不同路径的对比，**不是本轮优化前后变化**。尤其 Python 数学参考由许多运算、临时 Tensor 和循环组成，不是官方优化 kernel，不能用其结果推算生产收益。
- Dense 的 SDPA 连续 KV 和分页输入不是同一布局；报告分别计量转换。CUDA/Triton Split-4 使用预分配工作区，而 Auto 路径自行分配。MLA/Sparse 的 PyTorch 参考使用连续 KV，Sparse 参考计算全长并使用预吸收 Query；算子的分页读取、投影和实际计算工作不同，不能把比值解释为同工作量加速。
- 安装后完整测试为 **146 passed、0 skipped**。本次测试不验证真实 DeepSeek/Qwen 模型权重端到端输出。

原始记录： [Dense 第一次](../results/all_operators_20261029/dense_run1.json)、[Dense 第二次](../results/all_operators_20261029/dense_run2.json)、[阶段三至五第一次](../results/all_operators_20261029/stage345_single_run1.json)、[阶段三至五第二次](../results/all_operators_20261029/stage345_single_run2.json)。早先多 Query MLA 探索亦保留在 [实验一](../results/all_operators_20261029/stage345_run1.json) 和 [实验二](../results/all_operators_20261029/stage345_run2.json)，其 `mla_decode`/`sparse_mla` 使用 T=4，脚本已改用 T=1，故不与下表混合。

## 3. Dense GQA Decode

BF16，Hq=32、Hkv=8、D=128、page=16；Split 使用 4 段。括号为相对同一行的 CUDA 顺序基线变化。

| B | L | CUDA 顺序 µs | CUDA Split-4 µs（变化） | Triton Split-4 µs（变化） | Auto µs（变化） | SDPA 连续 µs | Gather+SDPA µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 88.5 | 40.6 (-54.1%) | 90.2 (+1.8%) | 88.7 (+0.2%) | 46.0 | 234.8 |
| 1 | 1024 | 672.4 | 188.1 (-72.0%) | 95.1 (-85.9%) | 124.6 (-81.5%) | 59.6 | 242.1 |
| 1 | 4096 | 2890.8 | 749.2 (-74.1%) | 116.8 (-96.0%) | 213.5 (-92.6%) | 59.0 | 245.3 |
| 4 | 128 | 95.6 | 61.6 (-35.6%) | 89.3 (-6.6%) | 94.9 (-0.7%) | 49.9 | 232.1 |
| 4 | 1024 | 733.3 | 411.1 (-43.9%) | 105.1 (-85.7%) | 120.2 (-83.6%) | 58.2 | 246.2 |
| 4 | 4096 | 4415.8 | 2278.4 (-48.4%) | 360.3 (-91.8%) | 452.5 (-89.8%) | 185.9 | 1014.4 |
| 16 | 128 | 207.3 | 187.1 (-9.8%) | 91.7 (-55.8%) | 208.7 (+0.7%) | 48.9 | 234.8 |
| 16 | 1024 | 2268.5 | 1749.8 (-22.9%) | 348.3 (-84.6%) | 364.7 (-83.9%) | 284.3 | 1118.6 |
| 16 | 4096 | 8985.3 | 7155.1 (-20.4%) | 1417.3 (-84.2%) | 1456.5 (-83.8%) | 1075.2 | 4358.9 |

`paged_gather` 单独计时也保存于 JSON。SDPA 连续列使用预先 Gather 的 KV；Gather+SDPA 才包含所用的布局转换，但仍不是完整推理框架端到端性能。

### 同条件历史复测：变化不等于优化收益

下表比较此前 `8c96ce1` 和本次 `0096124` 的两进程中位数：BF16、Hq=32、Hkv=8、D=128、P=16，测试脚本 SHA-256 均为 `5eb268dd609e098a8dc305759b07332142521bb4dd35366c7045e7bdd48600bb`，预热与采样参数相同。两次版本的 Dense CUDA 和 Triton 算子源码没有变化，但扩展新增了其他源码、分别构建，GPU 未隔离且未锁频。此表描述观测波动，不能归因于代码性能优化。

| B | L | 路径 | 先前 µs | 本次 µs | 本次相对先前 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 128 | CUDA 顺序 | 88.4 | 88.5 | +0.2% |
| 1 | 128 | Triton Split-4 | 92.3 | 90.2 | -2.3% |
| 1 | 1024 | CUDA 顺序 | 667.0 | 672.4 | +0.8% |
| 1 | 1024 | Triton Split-4 | 94.1 | 95.2 | +1.1% |
| 1 | 4096 | CUDA 顺序 | 2956.2 | 2890.8 | -2.2% |
| 1 | 4096 | Triton Split-4 | 118.1 | 116.8 | -1.1% |
| 16 | 4096 | CUDA 顺序 | 8883.1 | 8985.3 | +1.2% |
| 16 | 4096 | Triton Split-4 | 1399.0 | 1417.3 | +1.3% |

原始历史数据见 [第一次](../results/operator_comparison_20261029/results.json)、[第二次](../results/operator_comparison_20261029/repeat.json)。

## 4. Dense GQA Prefill

BF16，B=1、Hq=8、Hkv=2、D=64、等长 causal。PyTorch SDPA 与 FlashAttention 为本机同输入、同输出语义的库实现，不是厂商针对本 GPU 发布的固定性能保证。

| L | CUDA SIMT µs | Triton Tensor Core µs（对 CUDA） | PyTorch SDPA µs | FlashAttention 2.8.3 µs |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 82.8 | 59.4 (-28.2%) | 38.8 | 109.3 |
| 512 | 970.0 | 59.7 (-93.8%) | 41.5 | 108.5 |
| 1024 | 3687.3 | 117.0 (-96.8%) | 62.0 | 110.6 |

## 5. Paged KV Append

BF16，Hkv=8、D=128、P=16，每序列单 Token。PyTorch 对照预计算了 page/slot 索引，计时只包含两次索引写入；CUDA 路径使用 block table 和 positions。

| Batch | PyTorch 索引写 µs | CUDA Append µs | CUDA 相对 PyTorch |
| ---: | ---: | ---: | ---: |
| 1 | 40.8 | 15.3 | -62.4% |
| 4 | 58.6 | 22.0 | -62.5% |
| 16 | 40.5 | 19.4 | -52.0% |
| 64 | 38.4 | 14.7 | -61.7% |

## 6. MLA、Qwen、DSA 与配套算子

以下均为 **FP32**。每个算子的形状、精确对照说明和每组原始采样见 JSON。MLA 测试 `B=1,H=4,L=64,Dn=16,Dr=8,Dc=32,Dv=16`；Direct Prefill 与 latent attention 使用 T=4，Decode 与 Sparse MLA 使用 T=1，Sparse K=16。Qwen/DSA 尺寸见下表。PyTorch 数学参考是对照，不是官方优化内核。

| 算子 | 固定形状补充 | CUDA / 算子 µs | PyTorch 数学参考 µs | 相对参考变化 |
| --- | --- | ---: | ---: | ---: |
| `latent_kv_write` | N=64, Dc=32, Dr=8 | 27.9 | 61.3 | -54.6% |
| `mla_query_absorb` | T=4 | 27.0 | 65.5 | -58.8% |
| `mla_value_project` | T=4 | 22.2 | 66.0 | -66.4% |
| `mla_latent_attention` | T=4 | 53.6 | 343.1 | -84.4% |
| `mla_prefill` | T=4 | 128.5 | 722.0 | -82.2% |
| `mla_decode` | T=1 | 67.2 | 734.0 | -90.8% |
| `sparse_mla` | T=1, K=16 | 59.4 | 400.4 | -85.2% |
| `gated_attention` | B=2,T=16,H=4,Dv=32 | 22.4 | 40.6 | -44.7% |
| `delta_output` | B=2,T=16,H=4,Dv=32 | 33.3 | 148.3 | -77.6% |
| `causal_conv1d` | B=2,T=16,C=64,K=4 | 24.7 | 1799.1 | -98.6% |
| `causal_conv1d_update` | B=2,T=1,C=64,K=4 | 43.8 | 196.9 | -77.8% |
| `delta_recurrent` | B=2,T=1,H=4,Dk=Dv=16 | 25.1 | 508.5 | -95.1% |
| `delta_chunkwise` | B=2,T=16,H=4,Dk=Dv=16,C=16 | 750.2 | 8518.4 | -91.2% |
| `lightning_indexer` | B=1,T=4,Hi=4,Di=16,L=64 | 26.9 | 348.4 | -92.3% |
| `topk_select` | B=1,T=4,L=64,K=16 | 25.2 | 482.7 | -94.8% |

具体参考路径的边界：`latent_kv_write` 对照包含 `cat` 和索引写；矩阵投影对照为 `einsum`；`mla_prefill` 对照为连续 Direct 数学实现；`mla_decode` 对照为连续 Direct，而本算子使用分页 Absorption；`sparse_mla` 对照预先吸收 Query 并计算全长 masked score。Conv、Delta 的 PyTorch 参考按 Token 循环，Top-K 对照采用 stable argsort；这些参考的算子分解与 kernel 数不同。

## 7. 理论计算与硬件上界：用于分析，不能直接当作实测延迟

| 类别 | 理论工作量 / 信息量（忽略常数及 Softmax 等额外项） | 本机数据应如何解读 |
| --- | --- | --- |
| Dense Decode | QK 与 PV 约 `O(B Hq L D)`；分页 KV 的有效数据量约 `2 B L Hkv D × bytes` | GQA 中不同 Q head 共享 KV，实际 DRAM 读取次数受 cache 与实现影响；不能将存储字节直接当作测得带宽。 |
| Dense Prefill | 方形序列注意力主计算约 `O(B Hq T² D)`；causal 减少有效位置但不同实现可能计算遮挡 Tile | FlashAttention 避免物化完整 score，理论 IO 优势取决于分块；上表有本机 SDPA/FlashAttention 实测。 |
| MLA | 查询吸收约 `O(B T H Dn Dc)`，latent attention 约 `O(B T H L (Dc+Dr))`，Value 投影约 `O(B T H Dv Dc)` | 连续 Direct 与分页 Absorbed 的张量布局、临时分配、求值顺序不同；上表参考时间不可直接说明模型级压缩收益。 |
| Latent KV 存储 | 假设完整多头 KV 每 Token 是 `H(Dn+Dr+Dv)` 元素，latent 表示为 `Dc+Dr` 元素 | 测试 MLA 配置 H=4、Dn=16、Dr=8、Dv=16、Dc=32，FP32 原始表示为 640 B/Token，latent 为 160 B/Token，理论存储量比为 4:1；不含页元数据、权重与实际访问次数。 |
| Qwen Delta | 递归一步状态更新约 `O(B H Dk Dv)`；本 Chunkwise 有下三角计算，工作量依赖 chunk 和实现 | 本实现每 Head 单线程且 Chunkwise 为正确性基线；上表参考用 PyTorch 顺序循环，无法代表官方并行实现。 |
| DSA | Indexer 点积约 `O(B T Hi L Di)`；当前 Top-K 顺序实现约 `O(B T K L)`；Sparse MLA 只遍历选择的 K 个位置 | K=16、L=64 时选中位置是 25%；它不等于整体延迟节省 75%，因为 Indexer、Top-K、投影与读取均有成本。 |

[NVIDIA RTX 50 系列规格](https://www.nvidia.com/en-au/geforce/10-series/10-series-specs/)给出 RTX 5060 标称显存带宽 **448 GB/s**，这是硬件规格、不是实际测得吞吐。举例：若 FP32 latent 在 L=64 时刚好从显存各读取一次 64×160=10,240 B，且假设达到峰值带宽，传输量/带宽约 **0.023 µs**；这是极度理想化的流量模型，省略请求、索引、写回、权重、重复读取、Launch 与计算，也不能作为缓存命中场景的严格下界，更不能把上表延迟除以它宣称利用率。Nsight Compute 此前因 `ERR_NVGPUCTRPERM` 无法读取计数器，故本报告没有实测带宽或 occupancy。

## 8. 官方实际性能与可比范围

| 官方来源 | 官方披露的实际性能 / 条件 | 与本机的关系 |
| --- | --- | --- |
| [DeepSeek FlashMLA 固定版本](https://github.com/deepseek-ai/FlashMLA/blob/1aef31d163f11c0f2c7329fcda71d2985bfb47a0/README.md) | H800 SXM5、CUDA 12.8：内存受限配置最高 3000 GB/s、计算受限配置最高 580 TFLOPS；BF16/FP16 分页 KV，block size 64 | 不同 GPU、形状、指标和内核；没有本机同条件百分比。 |
| [DeepSeek Sparse FlashMLA 技术报告](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md) | H800 SXM5，FP8 KV / BF16 运算，B=128、Heads=128、Q=2、Top-K=2048 的计算受限配置达到 410 TFLOPS | 本机 FP32 小形状且未使用 FP8；不能据此计算加速比。 |
| [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention) | 提供 A100/H100 的 forward+backward 对照图，具体效果取决于序列长度和硬件 | 本次 §4 是在 RTX 5060 本机调用 FlashAttention 2.8.3 的 forward 实测；两者测试范围不同。 |
| [nano-vLLM 固定版本 README](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/README.md) | RTX 4070 Laptop、Qwen3-0.6B、256 随机长度序列：nano-vLLM 1434.13 token/s，vLLM 1361.84 token/s | 这是端到端引擎吞吐，不是本库的单算子延迟或 RTX 5060 数值。 |

对 `gated_attention`、卷积、Delta Recurrent/Chunkwise、Indexer、Top-K 和各辅助投影，没有找到与本报告**同 GPU、同 dtype、同张量契约与形状**的可直接比较官方 µs 数字，因此不编造官方成绩。上表 H800 的 GB/s、TFLOPS 和本机 operator µs 属于不同指标，也不应计算简单比例。

## 9. 后续优化思路（本轮未实施）

- Dense Decode：测量分段数、页容量与实际长度之间的关系，评估 KV Head 复用、连续访存、缓冲复用及合并 Launch 成本；Auto 的静态容量启发式尚未全面调参。
- Dense Prefill：比较 Tile/Warp 组合与遮挡 Tile 跳过；利用计数器确认 MMA 活动和资源压力后再修改。
- MLA：先分析 Direct 每 Token 重算投影、Absorbed 的中间 Tensor 舍入与分配；探索按 Head 批量投影、向量化分页读取和缓冲复用，并保持位置与 RoPE 语义。
- Qwen：对 Chunkwise 当前单线程基线，可评估分块矩阵并行、并行下三角求解；Recurrent/Conv 考虑状态访存和解码调用成本，保持 FP32 state 与更新顺序。
- DSA：Top-K 当前逐行 `O(KL)`，可研究稳定并列规则下的分段选择/CUB；Indexer 可研究分块与权重复用，Sparse MLA 需要把索引、选择和主计算的总成本一起测量。
- 以上是待验证假设，未实施。先解决 GPU 性能计数器权限，再通过 Nsight、长短上下文、不同 Batch/dtype 和重复实验确认瓶颈。

## 10. 复现

构建扩展后，使用已安装 Triton 与 FlashAttention 的环境，并确认 GPU 上没有干扰任务：

```bash
cd /home/ubuntu/ai-infra/minivllm/myvllm/atttention-kernel
/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python -m pytest -q -ra
/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python benchmarks/bench_all_operators.py --output results/NEW/dense1.json
/home/ubuntu/miniconda3/envs/ai-minisgl/bin/python benchmarks/bench_stage345.py --output results/NEW/stage345_1.json
```

每个命令重复一次并使用新的输出文件。各脚本保存计时样本、形状、环境和 Git HEAD；保存时工作区包含尚未跟踪的测试脚本。第二轮阶段三至五脚本 SHA-256：`27e62ea51ce45ade98edd9e17b2c7fbcb02cf70f097f0f3a8b10db92736ba0eb`。模型输入、缓存和跨 Stream 同步责任见算子契约。
