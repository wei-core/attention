# gqa_decode 算子契约 (v0.1)

输入（所有 Tensor 在同一个 CUDA device，连续布局）：

| 输入 | Shape | dtype |
|---|---|---|
| `q` | `[B, Hq, D]` | float32 / float16 / bfloat16 |
| `k_cache` | `[Nblocks, BlockSize, Hkv, D]` | 同 q |
| `v_cache` | `[Nblocks, BlockSize, Hkv, D]` | 同 q |
| `block_table` | `[B, MaxPages]` | int32 |
| `context_lens` | `[B]` | int32 |
| `scale` | scalar，默认 `1/sqrt(D)` | float |

输出：`out` `[B, Hq, D]`，与 q 相同 dtype，不修改任何输入。

逻辑头映射：`kv_head = q_head // (Hq / Hkv)`（Hq % Hkv == 0）。

逻辑 token `t` 在物理 KV Cache 中的位置：

```text
page = t // BlockSize
slot = t % BlockSize
physical_block = block_table[b, page]
k/v = cache[physical_block, slot, kv_head, :]
```

对于所有 t < context_lens[b] 执行稳定 Softmax：

```text
scores = (Q @ K^T) * scale
P = softmax(scores, dim=-1)
out = P @ V
```

- 当某个样本 `context_len == 0` 时，对应 `out` 为全 0。
- `context_lens[b] <= MaxPages * BlockSize`。
- 有效 block table 项必须在 `[0, Nblocks)`；无效尾部页可填 -1。
- 本版本不支持 head_dim > 256，不支持 strided tensor，不支持 K/V 与 Q 混合 dtype。
- 这是只读 paged single-token decode。未进行缓存追加、RoPE、Sliding Window、量化或状态更新。
- 包装层只在 CPU 检查 shape/dtype/device；不会回读 CUDA 上的 `context_lens`，以免引入隐藏的同步。GPU 元数据必须由调用方保证合法。

## Split-KV 与 KV Append（实验接口）

`gqa_decode_split` 接受与 `gqa_decode` 相同的输入，并通过 `splits` 指定 1–65535 个分段。每段生成 FP32 最大值、指数和和未归一化输出，再在同一 CUDA stream 合并；空段具有零权重。可传入预分配的 FP32 `workspace`，形状为 `[B,Hq,splits,D+2]`，由调用方负责并发调用之间的独占与生命周期；不传时每次调用分配临时 buffer。尚无自适应分段策略。

`kv_append(key, value, k_cache, v_cache, block_table, positions)` 原地写入每个序列的一个 token：`key/value` 为 `[B,Hkv,D]`，`positions` 为 CUDA int32 `[B]`，页布局与 decode 相同。调用方必须保证位置和物理页有效、不同写入不冲突，并负责分配页面、更新长度以及跨 stream 的同步。算子不检查 GPU 元数据值域，也不管理页面所有权。

可在调试时调用 `qd_attention.validation.validate_paged_metadata` 检查长度、页号与 append 冲突。该函数会同步并复制 GPU 元数据，不用于性能敏感路径。

`gqa_decode_split(..., backend="triton")` 使用按 Token 并行的分页读取和 FP32 合并，支持 1–64 个分段，workspace 布局相同。`gqa_decode_auto` 根据页容量、Batch、Head 和 GPU SM 数量估计分段数，无需同步读取 GPU 长度；当有效长度远小于页容量时可能不是最优。Triton 是可选依赖。workspace 不得与输入、输出或其他并发调用共享存储。
