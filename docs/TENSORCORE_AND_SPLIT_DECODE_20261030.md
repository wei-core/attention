# Delta 和 MLA Tensor Core 路径与分段 Decode 融合

本次基线为 `e5bd77f`。分别实现 Delta 的 WMMA 分块更新、MLA Prefill 的 WMMA QK/PV，以及长上下文 MLA Decode 的投影融合。RTX 5060 上前两条 WMMA 原型在本报告的形状中慢于现有 CUDA 路径，因此需要显式指定 `backend="wmma"`；默认仍使用原 CUDA 路径。Decode 分段融合默认启用。

## 修改思路和源码

| 模块 | 原始文件（e5bd77f 完整快照） | 修改后文件 | 实现 |
| --- | --- | --- | --- |
| Delta | [原始文件](source_history/e5bd77f/csrc/qwen/delta_chunkwise.cu.txt) | [修改后文件](../csrc/qwen/delta_chunkwise.cu) | 8 行对角块仍按依赖顺序解；后续行的矩阵更新使用 WMMA TF32 16×16×8。A、X 拆成高低 TF32 分量，累计高×高、高×低、低×高三次乘法并用 FP32 累积，忽略低×低项。 |
| MLA Prefill | [原始文件](source_history/e5bd77f/csrc/mla/mla_prefill.cu.txt) | [修改后文件](../csrc/mla/mla_prefill.cu) | 每 CTA 处理 16 Query、每 Tile 16 latent token；使用 FP16/BF16 WMMA 做 QK 和概率×latent，FP32 在线 Softmax、累积和 Value 投影。吸收后的 Query 与概率在 MMA 前转回输入精度。 |
| MLA Decode | [原始文件](source_history/e5bd77f/csrc/mla/mla.cu.txt) | [修改后文件](../csrc/mla/mla.cu) | 长上下文 4 个分段各自融合 Query Absorption 和 Attention，归并 kernel 直接融合 Value Projection，避免完整 Query/latent 中间 Tensor；保留分段 partial 工作区。 |
| 接口注册 | [原始文件](source_history/e5bd77f/csrc/binding/torch_ops.cpp.txt) | [修改后文件](../csrc/binding/torch_ops.cpp) | 为两个前向算子增加默认关闭的 use_mma 参数，原调用兼容。 |
| MLA API | [原始文件](source_history/e5bd77f/qd_attention/mla.py.txt) | [修改后文件](../qd_attention/mla.py) | Prefill 支持显式 wmma 后端；单 Query 长 dense / 大 sparse 候选也可进入融合分段 Decode。 |
| Delta API | [原始文件](source_history/e5bd77f/qd_attention/qwen.py.txt) | [修改后文件](../qd_attention/qwen.py) | 支持显式 wmma 后端，默认保留 CUDA。 |

快照通过 `git show e5bd77f:<path>` 获得，保留 [SHA-256](source_history/e5bd77f/SHA256SUMS)。快照内容与原文件逐字节一致；附加 `.txt` 是为了文档浏览。

Delta 的 WMMA **只承担三角系统求解中的块更新矩阵乘**，对角块前向代入仍是 SIMT，不能称为整个三角求解由 Tensor Core 执行。低×低项及浮点计算顺序可能带来误差。`backend="wmma"` 要求 compute capability ≥8.0、chunk 为 32 或 64、Dk+Dv≥32；没有覆盖其他架构的实测。MLA Prefill 的 WMMA 限 FP16/BF16、T≥16、Dc≤256、compute capability ≥8.0。条件不满足时显式后端会报错；默认后端支持原有范围。

## 性能对照

GPU：RTX 5060 8 GiB。旧代码 `e5bd77f` 与当前代码在相同环境运行，顺序为旧→新默认→新 WMMA→新 WMMA→新默认→旧，各为独立进程。每个形状预热 5 次、7 组 CUDA Event、每组 10 次调用；表值为两进程中位数的中位数。分配、启动和可能的 Python 提交间隙包含在内，GPU 未隔离、未锁频，不是 kernel-only 时间或生产服务吞吐。旧版和新默认的前两行采用普通 CUDA 路径，新 WMMA 列显式选择实验后端；Decode 在新默认与 WMMA 两列使用相同的分段融合路径。

| 路径 | 形状 | 旧版 μs | 新默认 μs | 新 WMMA μs | 新 WMMA 相对旧版 |
| --- | --- | ---: | ---: | ---: | ---: |
| delta_chunkwise | B=2, T=32, H=4, D=16, chunk=32, dtype=fp32 | 62.53 | 61.46 | 68.73 | +9.9% |
| delta_chunkwise | B=2, T=32, H=4, D=64, chunk=32, dtype=fp32 | 472.39 | 470.56 | 470.73 | -0.4% |
| delta_chunkwise | B=2, T=64, H=4, D=16, chunk=64, dtype=fp32 | 170.06 | 170.48 | 176.03 | +3.5% |
| delta_chunkwise | B=2, T=64, H=4, D=64, chunk=64, dtype=fp32 | 1730.33 | 1710.62 | 1740.28 | +0.6% |
| mla_prefill | B=1, T=16, H=4, L=64, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 51.89 | 51.44 | 107.27 | +106.7% |
| mla_prefill | B=1, T=16, H=4, L=256, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 172.32 | 174.23 | 253.10 | +46.9% |
| mla_prefill | B=1, T=64, H=4, L=64, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 56.75 | 58.41 | 104.65 | +84.4% |
| mla_prefill | B=1, T=64, H=4, L=256, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 187.94 | 189.91 | 246.73 | +31.3% |
| mla_decode | B=1, T=1, H=4, L=512, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 118.55 | 103.12 | 102.85 | -13.2% |
| mla_decode | B=1, T=1, H=4, L=1024, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 213.18 | 191.98 | 191.62 | -10.1% |
| mla_decode | B=1, T=1, H=4, L=4096, Dn=32, Dr=16, Dc=64, Dv=32, dtype=bf16 | 789.89 | 736.50 | 737.01 | -6.7% |

原始数据：[旧 1](../results/tensorcore_followup_20261030/before1.json)、[旧 2](../results/tensorcore_followup_20261030/before2.json)、[新默认 1](../results/tensorcore_followup_20261030/default1.json)、[新默认 2](../results/tensorcore_followup_20261030/default2.json)、[WMMA 1](../results/tensorcore_followup_20261030/after1.json)、[WMMA 2](../results/tensorcore_followup_20261030/after2.json)。长 Decode 的改善受序列、分页、维度和负载影响；不能据此推断完整模型收益。

## 精度与指标边界

完整 CUDA 回归 **213 passed、0 skipped**。增加了长 Decode 的 FP32/FP16/BF16、dense/sparse、causal/noncausal、空上下文与非完整分段测试；Delta 对不同 chunk、输入 dtype、非整矩阵维度及状态；Prefill 对 FP16/BF16、非整维度、变长、空行和掩码测试。未进行完整模型权重或反向传播验证。

三个随机种子的额外对照（WMMA 对当前 CUDA 默认）：Delta FP32 输出最大绝对误差不超过 5.97e−8、最终状态不超过 4.77e−7；MLA 输出最大绝对误差 FP16 为 0.000977、BF16 为 0.007813。形状及全部样本见 [error.json](../results/tensorcore_followup_20261030/error.json)，这些范围不是所有输入的数值误差上界。

示例：MLA Prefill、B=1、T=64、H=4、L=256、Dn=32、Dr=16、Dc=64、Dv=32，末端对齐 causal。有效长度之和为 14,368，仅统计吸收投影、有效 QK、加权 latent 和 Value 投影，得到 18,649,088 有效 FLOP。假设唯一 Q、K、权重及输出各读写一次，理想逻辑字节 114,948 B，模型算术强度约 162.24 FLOP/B。此工作量忽略 Softmax、掩码、填充与真实 WMMA 指令的额外工作。
该形状新默认约 0.0982 有效 TFLOPS，WMMA 约 0.0756 有效 TFLOPS；两者均为有效数学工作除以算子事件时间，不是硬件实际 FLOP 吞吐。

标称显存带宽乘逻辑算术强度只是示意带宽项，不能用来计算真实 Roofline 效率。此次重新尝试 Nsight Compute 返回 [ERR_NVGPUCTRPERM](../results/tensorcore_followup_20261030/ncu_attempt.log)，无法采集实际 DRAM 字节与硬件 FLOP，也不能宣称实测 Tensor Core 利用率或性能瓶颈。启用宿主机性能计数器后，应按 kernel 采集 Roofline 数据。

复现：编译旧提交与当前源码后，使用 `python benchmarks/bench_tensorcore_followup.py --root /path/to/e5bd77f --output before.json`、`python benchmarks/bench_tensorcore_followup.py --root "$PWD" --output default.json` 和 `python benchmarks/bench_tensorcore_followup.py --root "$PWD" --wmma --output wmma.json`；`python benchmarks/check_tensorcore_error.py` 重算误差样本，`python -m pytest -q` 执行正确性回归。
