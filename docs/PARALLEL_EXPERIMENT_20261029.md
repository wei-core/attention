> 历史中间实验；后续实现及重新对照见 [优化报告](ATTENTION_OPTIMIZATION_20261029.md)。

# Delta 与 DSA 并行化实验（未完成）

此实验是未完成的优化工作，不能视作完整的阶段目标或生产性能验收。代码基于 `de89326`，仅修改 Delta Chunkwise、Indexer 与 Top-K。GQA CUDA Prefill、MLA Prefill/Decode、其他轻量算子和端到端 DSA 测试尚未完成。

- Delta Chunkwise：一个 CTA 负责一个 head，特征规约使用 Warp shuffle 和共享内存，输出列和状态元素由线程分工；下三角前向代入仍逐行逐项同步，临时矩阵仍存于全局内存，没有分块并行矩阵求解或 Tensor Core 路径。
- Indexer：一个 Warp 负责一个 Query-Key 分数，feature 由 lane 并行求和，head 仍逐个计算。
- Top-K：每行一个 CTA 并行扫描并归约最大值，按分数降序、索引升序保证确定性；仍对每个结果重复扫描，没有分块候选合并。

`python -m pytest -q` 在 RTX 5060、PyTorch/CUDA 环境中通过 152 项测试；覆盖多种 dtype、非整 Warp 维度、非有限值、分数相同及 K 大于候选数。性能实验采用 `benchmarks/bench_stage345.py --warmup 5 --samples 5 --inner 5`，GPU 未隔离、未锁频，CUDA Event 计时包括 Python 提交间隙和分配。以下是两次独立进程的中位数，原始样本见 [第 1 次](../results/parallel_experiment_20261029/experiment_run1.json) 与 [第 2 次](../results/parallel_experiment_20261029/experiment_run2.json)。两次都是修改后的实现；JSON 中的 git_head 指向基础提交，实际工作区有未提交更改。

| 算子和形状 | 第 1 次 μs | 第 2 次 μs |
| --- | ---: | ---: |
| Delta Chunkwise，B=2 T=16 H=4 Dk=Dv=16 C=16 | 108.74 | 75.02 |
| Indexer，B=1 T=4 H=4 D=16 L=64 | 11.42 | 14.42 |
| Top-K，B=1 T=4 L=64 K=16 | 18.04 | 55.79 |

历史版本和形状见 [原始报告](ATTENTION_METRICS_ROOFLINE.md)；历史与本次未采用交替同机基准，且本次波动显著，因此不能将差异称作可靠加速比。尚未测实际 DRAM 字节及执行 FLOP，不能报告实测 Roofline 效率；尤其 Delta 的等价递推 FLOP 并不是这个实现执行的 FLOP。下一步需要隔离 GPU 并交替运行两个版本、扩大形状覆盖、记录 kernel 级耗时，并在启用性能计数器后采集硬件 Roofline。
