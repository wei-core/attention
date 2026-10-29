# Attention 算子清单与性能对照

本报告对应独立算子库 QD Attention，不包含 Nano-vLLM 集成。测试于 2026-10-29（北京时间）在本机 RTX 5060 上完成。表中延迟越低越好；这是一组固定配置的微基准，不代表模型端到端吞吐或所有形状上的最佳结果。

## 1. 当前有哪些算子

| 公开接口 / 路径 | 分类 | 实际功能 | 当前状态 |
| --- | --- | --- | --- |
| `gqa_decode` | Attention | CUDA 单 Token Paged Decode，顺序扫描 KV；MHA/MQA/GQA 共用头映射 | 已实现，作为正确性及性能基线 |
| `gqa_decode_split(..., backend="cuda")` | Attention | CUDA 将一条 KV 序列分段计算，再合并在线 Softmax 状态 | 已实现，支持显式 workspace |
| `gqa_decode_split(..., backend="triton")` | Attention | Triton 按 Token 并行读取分页 KV，分段计算并合并 | 已实现，支持显式 workspace |
| `gqa_decode_auto` | 调度接口 | 按分页容量、Batch、Head 和 SM 数量选择基线或 Triton Split-KV | 已实现的初步启发式；不是额外的数学算子 |
| `gqa_prefill(..., backend="cuda")` | Attention | Dense GQA Prefill，CUDA SIMT + 在线 Softmax；支持变长及右下对齐 Causal Mask | 已实现，正确性基线 |
| `gqa_prefill(..., backend="tensorcore")` | Attention | Triton 分块 Prefill，FP16/BF16 Tensor Core MMA、FP32 状态 | 已实现；SM120 PTX 中观察到 `mma.sync` |
| `kv_append` | KV Cache 配套 | 把每个序列一个 Token 的 K/V 写到分页缓存 | 已实现；不是 Attention 计算，但单独测量 |

`reference_gqa_decode` 和 `reference_gqa_prefill` 是正确性参考实现，调试校验函数不属于推理算子；在线 Softmax、归约和 Split 合并是内部组件，而非独立公开 Attention 接口。MHA、MQA、GQA 是上述接口中的 Head 配置，不是三套独立算子。MLA、Sparse、Linear、Compressed 目录目前为规划占位，没有可运行实现或可报告的性能数据；Backward、量化和 Nano-vLLM 集成也不在当前实现中。张量契约分别见 [Decode/KV](KERNEL_CONTRACT.md) 和 [Prefill](PREFILL_CONTRACT.md)。

## 2. 环境与方法

- GPU：RTX 5060 8 GiB，SM120；Windows 驱动 617.14，WSL 环境。测试开始时 Xwayland 占用显存；GPU 未隔离，时钟未锁定。
- 软件：PyTorch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1、FlashAttention 2.8.3、Python 3.12.14。被测算子源码为 Git 提交 `8c96ce1`；测试脚本本身当时为未跟踪文件。
- BF16 输入，随机种子 123；各路径先与等价对照做数值检查。两次独立进程顺序执行相同计时流程；每次每个路径预热 10 次，测量 5 组，每组连续调用 10 次。表中数值是**两次运行各自中位数的中位数**（两数的均值），单位为每次调用微秒（µs）。两轮固定执行顺序、未做随机化或 GPU 隔离，因此不推断统计显著性。
- 使用 CUDA Events 测整个算子调用区间，包含 kernel 启动、输出分配和可能的主机提交间隙；不是纯 Kernel 时间。Decode Split 对照预分配了 workspace；自动调度仍自行分配。相对变化为 `(被比较延迟 / 基线延迟 - 1) × 100%`，负值表示更快。由于短算子接近启动开销，百分比易受抖动影响。
- PyTorch SDPA 的连续 K/V 与分页算子布局不同；分别列出连续输入、分页 Gather，以及 Gather + SDPA 的调用。Prefill 为等长方形 causal 输入。`kv_append` 的 PyTorch 参考已预计算物理页与槽位，未把计算索引的开销计入，因而仅是有明确边界的参考对照。

原始数据与每组采样、环境信息： [首次运行](../results/operator_comparison_20261029/results.json)、[第二次运行](../results/operator_comparison_20261029/repeat.json)。首次运行对 CUDA Append 的写入结果做检查；第二次运行额外检查 PyTorch 参考写入。两次测量函数相同。测试脚本 SHA-256（第二次运行）：`5eb268dd609e098a8dc305759b07332142521bb4dd35366c7045e7bdd48600bb`。

## 3. Paged Decode：各路径性能

配置：Hq=32、Hkv=8、D=128、Page=16，BF16；CUDA/Triton Split 都使用 4 段及预分配 workspace。括号为相对同一行 CUDA 顺序基线的延迟变化。

| B | KV 长度 | CUDA 顺序基线 µs | CUDA Split-4 µs（变化） | Triton Split-4 µs（变化） | Auto µs（变化） | SDPA 连续 µs | Gather + SDPA µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 88.4 | 38.0 (-57.0%) | 92.3 (+4.5%) | 88.7 (+0.4%) | 48.4 | 231.7 |
| 1 | 1024 | 667.0 | 187.9 (-71.8%) | 94.1 (-85.9%) | 130.2 (-80.5%) | 64.6 | 262.1 |
| 1 | 4096 | 2956.2 | 715.4 (-75.8%) | 118.1 (-96.0%) | 213.8 (-92.8%) | 69.7 | 242.2 |
| 4 | 128 | 96.8 | 61.3 (-36.6%) | 89.5 (-7.5%) | 104.3 (+7.7%) | 49.0 | 226.5 |
| 4 | 1024 | 741.7 | 408.3 (-45.0%) | 102.4 (-86.2%) | 126.0 (-83.0%) | 60.3 | 252.1 |
| 4 | 4096 | 4348.6 | 2245.7 (-48.4%) | 372.6 (-91.4%) | 445.4 (-89.8%) | 185.8 | 941.5 |
| 16 | 128 | 217.2 | 185.8 (-14.5%) | 88.3 (-59.4%) | 207.0 (-4.7%) | 49.1 | 242.3 |
| 16 | 1024 | 2242.5 | 1709.9 (-23.8%) | 354.0 (-84.2%) | 387.6 (-82.7%) | 282.2 | 1070.6 |
| 16 | 4096 | 8883.1 | 6974.8 (-21.5%) | 1399.0 (-84.3%) | 1471.2 (-83.4%) | 1112.4 | 4363.7 |

SDPA 连续输入不承担分页转换，不能直接等同于分页端到端路径；`paged_gather` 的独立结果也保存在 JSON 中。4 段只是本次固定对照，不代表所有负载的最优分段。历史的 1/2/4/8/16 段探索保留在 [阶段二原始对照](../results/stage1_20261029/stage2_comparison.json)。

**优化思路：** 在短序列和高 Batch 时减少额外合并 Kernel、分配与 Python 提交成本；在长序列按 GPU SM 数和实际有效长度选择 KV 分段；进一步测量同一 KV Head 对多个 Query Head 的数据复用、访问合并、寄存器与 occupancy。当前自动调度只看到分配容量而不读取 GPU 上的真实长度，因此变长或容量远大于实际长度时需要重新调参。性能计数器尚无数据，不能把某一瓶颈当作已证实结论。

## 4. GQA Prefill：各路径性能

配置：B=1、Hq=8、Hkv=2、D=64、BF16、等长 causal；括号为相对 CUDA SIMT 基线的变化。SDPA 和 FlashAttention 是外部实现与相应包装路径，包含各自的调用开销。

| Q/KV 长度 | CUDA SIMT µs | Triton Tensor Core µs（变化） | PyTorch SDPA µs（变化） | FlashAttention µs（变化） |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 81.0 | 65.0 (-19.8%) | 41.5 (-48.8%) | 112.7 (+39.1%) |
| 512 | 927.7 | 65.8 (-92.9%) | 39.7 (-95.7%) | 108.1 (-88.3%) |
| 1024 | 3655.1 | 121.3 (-96.7%) | 69.5 (-98.1%) | 106.8 (-97.1%) |

Tensor Core 路径对概率 Tile 转换到输入精度后执行 PV，可能与 FP32 SIMT 累加存在舍入差异；上表测量前检查了输出误差。PTX 指令说明该路径使用 MMA，不足以证明 Tensor Core 的实际利用率。

**优化思路：** 根据序列长度、Head Dimension、dtype 和硬件搜索 BM/BN/Warp 配置，避免短序列使用过大 Tile；评估跳过 causal 全遮挡 Tile、减少重复 K/V 读取以及复用中间缓冲；使用 Nsight Compute 验证 Tensor Core 活动、访存和寄存器压力后再决定具体改动。

## 5. KV Append：单 Token 写入

配置：Hkv=8、D=128、Page=16、BF16；每序列写入 1 个 Token。变化率以预计算索引的 PyTorch 写入为对照。

| Batch | PyTorch 两次索引写入 µs | CUDA Append µs | CUDA 相对变化 |
| ---: | ---: | ---: | ---: |
| 1 | 38.3 | 14.4 | -62.4% |
| 4 | 55.5 | 24.1 | -56.6% |
| 16 | 48.0 | 19.5 | -59.3% |
| 64 | 34.5 | 16.3 | -52.7% |

Batch 间并不保证单调：该测试的调用时间短，样本波动和 GPU 状态可能主导结果。物理页必须有效、写入目标不能冲突；页面生命周期及跨 Stream 顺序由调用方管理。

**优化思路：** 对齐条件满足时评估向量化读写、连续目标的批量写入和与后续 Decode 的融合；先对比不同 Batch 和页面布局，再检查访存带宽及 Launch 占比。不要仅根据单次小 Batch 的微秒差判断端到端收益。

## 6. 正确性、历史与尚需补充

- 此版本完整测试上次执行结果为 96 passed、0 skipped，覆盖三种 dtype、分页、空序列、变长和边界；上表另在计时前对具体负载检查数值。测试不证明所有可能形状均无误。
- [早期基线](../results/baseline_v1/README.md)是在同步修复前且 GPU 有其他任务时得到的探索结果；[2026-10-29 阶段一和阶段二记录](../results/stage1_20261029/README.md)含三种 dtype 的 27 组基线和 Nsight Systems 原始报告。不同轮次、样本数和测试配置不能直接视为同条件的性能提升。
- Nsight Compute 2026.3.1 在普通用户和 `sudo` 下均返回 `ERR_NVGPUCTRPERM`，没有可用计数器报告；宿主机启用权限后应补测，不应推断实际带宽、occupancy 或 Tensor Core 利用率。Nsight Systems 的阶段一报告仅给出指定内核的时间线，见前述记录。
- 目前没有 CUDA Graph 下的 kernel-only 数据、全 dtype/Head Dimension 的性能矩阵、端到端吞吐或自动调度的全面调参；本表是 operator 路径的固定形状对照。

## 7. 复现

确保扩展已按项目 README 构建，安装相应版本的 Triton 和 FlashAttention；GPU 存在其他任务时先记录状态。运行：

```bash
cd /home/ubuntu/ai-infra/minivllm/myvllm/atttention-kernel
/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python -m pytest -q -ra
/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python benchmarks/bench_all_operators.py --output results/NEW_DIRECTORY/run1.json
/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python benchmarks/bench_all_operators.py --output results/NEW_DIRECTORY/run2.json
```

可通过 `--warmup`、`--repeats`、`--inner` 调整采样次数；如果 FlashAttention 或 GPU 不可用，脚本会报错，不会静默把对照当作已完成。

