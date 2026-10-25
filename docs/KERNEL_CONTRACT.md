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
