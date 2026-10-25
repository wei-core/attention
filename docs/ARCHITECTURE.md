# 算子库架构 v0.1

```text
                    Python API (qd_attention/)
                              |
                  PyTorch C++ operator binding
                              |
                   CUDA launch / param checks
                              |
       +-----------+----------+-----------+------------+
       |           |          |           |            |
     dense        mla       linear      sparse      compressed
  [implemented] [planned]   [planned]   [planned]     [planned]
       +-----------+----------+-----------+------------+
                              |
                      common / cache
                       CUDA primitives
                              |
                        NVIDIA GPU
```

## 模块边界

- `csrc/dense/`: MHA/MQA/GQA 的注意力核心计算；已实现 paged single-token decode 基线。
- `csrc/mla/`: 独立的 MLA 数学结构与压缩 KV Cache 表示；不能靠 Dense 的头数参数代替。
- `csrc/linear/`: Gated DeltaNet / 状态递推，区别于 Softmax Attention。
- `csrc/sparse/`: Indexer/Top-K/稀疏注意力计算，后续可复用 MLA 基础设施。
- `csrc/compressed/`: KV 压缩、压缩稀疏注意力，按模型具体版本设计。
- `csrc/common/`: 低层数值与并行辅助，不引入模型及框架依赖。
- `csrc/binding/`: `TORCH_LIBRARY` 注册和 Python/C++ 桥接。
- `qd_attention/reference/`: 正确性 oracle；不依赖 CUDA 扩展编译成功。
- `tests/`: 正确性、边界输入、数据类型与形状校验。
- `benchmarks/`: 形状扫描、Kernel Latency；未来增加 Nsight 性能案例。

## 设计原则

1. **算子优先**：不依赖模型加载器、scheduler、Block Manager、nano-vLLM。
2. **数学路径分离**：Dense / MLA / Linear 不强求同一个底层 kernel。
3. **公共契约先行**：每个 kernel 明确布局、dtype、mask、返回值和状态副作用。
4. **Reference + Tests 先于性能优化**：没有可复现对照，不做“更快”的结论。
5. **暂不做一体化全能 Dispatcher**：首个版本 API 直接指定 gqa_decode，避免调度层过度设计。
6. **GPU 可移植性**：针对 SM120 进行性能优化时，按真实资源限制重选 Tile/warp 组合。

## 下一批可实施 Kernel

按依赖顺序建议：

```text
gqa_decode (current)
   -> gqa_prefill (FlashAttention-style)
   -> split_kv_decode / paged_prefill
   -> mla_decode (Latent KV + RoPE separation)
   -> gated_delta_recurrent / chunkwise
   -> dsa_indexer + sparse_mla
```

未来可增加 Runtime Dispatch，根据 head_dim、batch_size、seq_len、dtype 决定内核变体；当下无需为不曾实现的路径构造虚假入口。
