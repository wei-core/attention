# 阶段三至五：正确性验证记录

2026-10-29（Asia/Shanghai），在 NVIDIA GeForce RTX 5060（SM120）、CUDA Toolkit 12.8、PyTorch 2.9.1+cu128 环境构建 CUDA 扩展，执行 `python -m pytest -q -ra`：**146 passed，0 skipped**。其中 50 项为本轮 MLA、Qwen 和 Sparse 专项测试，96 项为既有测试。

专项覆盖 Direct 与 Absorbed MLA 的数学对照、FP32/FP16/BF16、错序物理页、长度和绝对位置、空上下文与 `Dc=512`；Qwen 中验证逐特征 Gate、FP32 状态单步递归、不同 Chunk Size/多 Chunk、卷积 Prefill 与 Decode 状态一致性、Gated RMSNorm；DSA 中验证 Indexer 公式、Top-K 并列和不足 K 的填充，以及索引到 Paged Sparse MLA 的完整路径。

这些测试使用随机或构造输入与独立 PyTorch 数学参考，不是 DeepSeek/Qwen 真实模型权重、完整模型输出或框架集成验证。低精度投影会改变舍入顺序。未进行模型精度、性能测试、内核调参、Nsight 或 FP8；Chunkwise 与 Top-K 为正确性优先的串行基线。GPU 元数据由调用方保证有效。详细契约见 [STAGE345_CONTRACT.md](STAGE345_CONTRACT.md)。

```bash
export CUDA_HOME=/home/ubuntu/miniconda3/envs/ai-minisgl
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
# 本机 Conda Toolkit 的 cuBLAS/cuSPARSE headers 来自 PyTorch 的 NVIDIA 依赖包：
export CPATH=$(find "$CUDA_HOME/lib/python3.12/site-packages/nvidia" -type d -name include | paste -sd:)
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q -ra
```
