# CUDA Attention 并行化与性能对照

本报告对比基线 `de89326` 和本次工作区的 CUDA 实现。基线从该提交独立导出并重新编译，两边使用相同 Python、CUDA、GPU 与驱动。运行顺序为新→旧→旧→新，每次独立进程；每组预热 5 次、CUDA Event 7 组，每组 10 次调用，表中取两次进程中位数的中位数。耗时含算子分配、启动及可能的 CPU 提交空隙，不是单 kernel 耗时。RTX 5060 8 GiB，GPU 未隔离、未锁频；对短算子的小差异不应解释为统计显著。

原始数据：[旧 1](../results/optimization_20261029/baseline1.json)、[旧 2](../results/optimization_20261029/baseline2.json)、[新 1](../results/optimization_20261029/new1.json)、[新 2](../results/optimization_20261029/new2.json)。完整参数及每组采样均在 JSON；所有结果为固定合成形状，不代表真实模型端到端吞吐。

| 算子 | 形状（未标注时 FP32） | 旧延迟 μs | 新延迟 μs | 变化 |
| --- | --- | ---: | ---: | ---: |
| gqa_prefill_cuda | L=128, H=8, Hkv=2, D=64, dtype=bf16 | 81.77 | 81.48 | -0.4% |
| gqa_prefill_triton | L=128, H=8, Hkv=2, D=64, dtype=bf16 | 58.46 | 60.95 | +4.3% |
| gqa_prefill_cuda | L=512, H=8, Hkv=2, D=64, dtype=bf16 | 919.93 | 607.76 | -33.9% |
| gqa_prefill_triton | L=512, H=8, Hkv=2, D=64, dtype=bf16 | 58.67 | 63.56 | +8.3% |
| gqa_prefill_cuda | L=1024, H=8, Hkv=2, D=64, dtype=bf16 | 3523.20 | 1917.68 | -45.6% |
| gqa_prefill_triton | L=1024, H=8, Hkv=2, D=64, dtype=bf16 | 117.24 | 115.66 | -1.4% |
| delta_chunkwise | B=2, T=32, H=4, D=16, chunk=16 | 1482.31 | 57.26 | -96.1% |
| delta_chunkwise | B=2, T=32, H=4, D=64, chunk=16 | 15985.67 | 248.55 | -98.4% |
| mla_prefill | B=1, T=4, H=4, L=64, Dn=16, Dr=8, Dc=32, Dv=16 | 124.88 | 44.78 | -64.1% |
| mla_decode | B=1, T=1, H=4, L=64, Dc=32, Dr=8 | 75.21 | 77.76 | +3.4% |
| indexer | B=1, T=1, L=64, H=4, D=16 | 18.71 | 18.84 | +0.7% |
| topk | B=1, T=1, L=64, K=16 | 20.66 | 20.06 | -2.9% |
| sparse_mla | B=1, T=1, L=64, K=16 | 58.21 | 62.07 | +6.6% |
| dsa_complete | B=1, T=1, L=64, K=16 | 93.72 | 87.82 | -6.3% |
| mla_prefill | B=1, T=4, H=4, L=1024, Dn=16, Dr=8, Dc=32, Dv=16 | 1934.80 | 613.79 | -68.3% |
| mla_decode | B=1, T=1, H=4, L=1024, Dc=32, Dr=8 | 763.24 | 209.51 | -72.5% |
| indexer | B=1, T=1, L=1024, H=4, D=16 | 20.70 | 18.14 | -12.4% |
| topk | B=1, T=1, L=1024, K=16 | 224.84 | 38.85 | -82.7% |
| sparse_mla | B=1, T=1, L=1024, K=16 | 64.45 | 55.92 | -13.2% |
| dsa_complete | B=1, T=1, L=1024, K=16 | 254.72 | 100.54 | -60.5% |
| mla_prefill_tiled | B=1, T=16, H=4, L=256, Dn=16, Dr=8, Dc=32, Dv=16 | 564.80 | 193.92 | -65.7% |
| gated_attention | B=2, T=256, H=4, D=64 | 18.55 | 18.16 | -2.1% |
| delta_output | B=2, T=256, H=4, D=64 | 20.20 | 19.32 | -4.3% |
| causal_conv1d | B=2, T=256, C=128, K=4 | 79.45 | 20.43 | -74.3% |
| kv_append | B=4, H=8, D=128, dtype=bf16 | 13.30 | 13.41 | +0.9% |

表中变化为 `新延迟/旧延迟−1`。Triton、Gate 和 Delta Output 未修改，差异主要反映测量波动，不能归因于本次优化。`dsa_complete` 包含 Indexer、Top-K、Query Absorption、Sparse Attention 和 Value Projection；它的时间不可通过将各单项中位数简单相加得到。

## 实现与边界

- **GQA Prefill**：FP16/BF16 且 Query 物理长度超过 128、GPU compute capability ≥ 8.0 时使用 CUDA WMMA：每 CTA 处理 16 个 Query，16 个 Key 为一块；Q/K/V 和概率放在共享内存，QK、PV 使用 WMMA，softmax 和输出归并用 FP32。因果块提前终止。较短 Query、FP32 或低架构继续使用旧 SIMT；概率在 PV 前转回输入精度，数值精度与 FP32 PV 不完全相同。当前单 Warp、同步加载、共享内存中间状态以及未跨 Query Head 复用 KV，远落后于现有 Triton Tensor Core 实现。
- **Delta Chunkwise**：并行构造下三角 Gram 矩阵；K tile 可放入共享内存；不同 RHS 列并行做前向代入，并行更新输出与状态。单列的三角依赖仍按顺序求解，Gram 是 SIMT，尚无分块矩阵求解或 MMA 路径。
- **MLA Prefill**：每个 Query 先做 Query Absorption，在 latent 空间做注意力后再执行 Value Projection，避免对每个 Query–Key 对重复投影；临时结果保持 FP32。T≥8、latent 宽度≤1024 且 RoPE 宽度≤256 时，一个 CTA 的四个 Query Warp 共享逐 token 加载的 latent/rope 数据；其余常见形状每 Query 使用一个 CTA，latent 宽度大于 4096 时回退原始 Direct 路径。当前每次缓存一个 latent token，尚未实现矩阵化多 token Latent KV Tile。
- **DSA**：Indexer 每个 Warp 并行归约特征，Head 仍逐个累加；L>256 且 K≤64 时 Top-K 每 256 个元素分块选候选并并行归并，其他情况使用逐次并行扫描。所有路径按分数降序、索引升序选有限值并用 −1 补齐。
- **MLA Decode**：静态容量或稀疏候选达到 512 时拆成四段，各段计算在线 Softmax 后归并；短上下文仍用原路径。Query 与 Value 投影仍是独立 kernel 并产生中间 Tensor，尚未融合。
- **其他算子**：Cache Append 对 16 字节对齐且 token 字节数为 16 倍数的输入采用向量复制，否则回退标量；Conv 在时间维按 16 token 分块，卷积、SiLU 与状态更新仍同一 kernel。Gate 与 RMSNorm/SiLU 本就分别融合在各自 kernel 内，未更改。

## 有效 TFLOPS 与 Roofline

以 BF16、B=1、Hq=8、Hkv=2、D=64、L=1024 的 causal GQA 为例，只统计有效位置 QK 和 PV 的乘加（乘加计 2 FLOP），有效工作量为 `4×8×64×1024×1025/2 = 1,074,790,400 FLOP`。忽略 mask tile、Softmax、指数、类型转换等执行工作，故得到的是有效 TFLOPS，不是硬件执行吞吐。理想逻辑字节假定 Q、唯一 K/V 和输出各读写一次，加两个 int32 长度，即 2,621,448 B，模型算术强度约 410 FLOP/B。
- `gqa_prefill_cuda`：1917.68 μs，约 0.560 有效 TFLOPS，约 533,980 Query/s。
- `gqa_prefill_triton`：115.66 μs，约 9.293 有效 TFLOPS，约 8,853,720 Query/s。

使用标称 448 GB/s 乘逻辑 AI 得到约 183.7 TFLOP/s 的**假设带宽项**；BF16 Tensor Core 的可持续计算上限未实测，不适用旧报告的 FP32 CUDA-core 19.2 TFLOPS 上限。逻辑字节不是实测 DRAM 流量，所以上述点与带宽项的比值不能叫硬件 Roofline 效率，也无法据此断言瓶颈。Nsight Compute 本次再次返回 `ERR_NVGPUCTRPERM`，详见 [日志](../results/optimization_20261029/ncu_attempt.log)；启用宿主机 NVIDIA GPU 性能计数器后，还需采集对应 kernel 的实际 FLOP、DRAM 字节及时间，再计算真实 Roofline。

## 正确性与复现

RTX 5060 上完整测试 `177 passed`，包括 FP16/BF16 WMMA 分块边界和大 logits、Delta 分块和跨 Warp 维度、长上下文 MLA 分段、Top-K 相同分数及非有限数、Cache 未对齐回退和 Conv 时间分块。没有完整模型权重和生产服务验证；GPU metadata 的合法性仍由调用方保证。

基线和当前版本分别编译后执行：

```bash
python benchmarks/bench_optimization.py --root /path/to/de89326 --output baseline.json
python benchmarks/bench_optimization.py --root "$PWD" --output optimized.json
python -m pytest -q
```

计数器启用方法和模型的局限见 [指标报告](ATTENTION_METRICS_ROOFLINE.md)。
