# 阶段三至五计算核心接口

本库仅提供 forward 数学核心。所有 CUDA 输入需连续、位于同一设备；浮点输入支持 FP32、FP16、BF16（状态及特殊输出另述）。均在 PyTorch 当前 CUDA Stream 执行，不提供 backward。性能、完整模型权重投影、RoPE、Cache 分配和 Nano-vLLM Adapter 均不在本轮范围。非 FP32 路径的中间量可能舍入，不能要求逐位一致。

## 阶段三：MLA

详见 [MLA 契约](MLA_CONTRACT.md)。`mla_prefill(q_nope,q_rope,latent,rope_key,w_uk,w_uv,lengths,positions,scale=...)` 使用连续 `[B,L,Dc]` latent 和 `[B,L,Dr]` rope key，直接计算 `q_nope·(W_UK latent)` 与位置得分，并对 `W_UV latent` 加权；所有 query 的因果边界是绝对 `positions[B,T]`，而非默认从零开始。空行输出零。Value 维上限 512；该实现是正确性基线，每个历史 Token 重算投影。

`mla_decode` 使用 `mla_query_absorb → mla_latent_attention → mla_value_project`。输入的 latent 已完成 RMSNorm，rope Q/K 已完成位置编码；scale 可由调用方指定。缓存写入按物理槽，读取按逻辑块表。投影后的中间 Tensor 以输入 dtype 存储，因此低精度与 Direct 路径可能有额外舍入。

## 阶段四：Qwen Hybrid

- `gated_attention(attn, gate)` 对形状与类型相同的逐特征原始 gate logits 应用 sigmoid，再与 Attention 输出相乘。
- `causal_conv1d(x, weight, state)` 的 x 为 `[B,T,C]`、weight 为 `[C,K]`、state 为 `[B,C,K]`（从旧到新）。每个 Token 左移窗口、追加输入、做 depthwise 点积和 SiLU；返回 `(y,new_state)`，不修改输入 state。`causal_conv1d_update` 使用单 Token `[B,C]` 调用同一核心。初始全零状态对应无历史前缀。
- `delta_recurrent(q,k,v,g,beta,state)` 输入分别为 `[B,H,Dk]`、`[B,H,Dk]`、`[B,H,Dv]`、`[B,H]`、`[B,H]`、`[B,H,Dk,Dv]`；g、beta、state 为 FP32，Q/K/V dtype 相同。g 为已经计算的 log decay（通常 `g<=0`）、beta 为已经 sigmoid 的门控（通常位于 `[0,1]`）；这些 GPU 值域由调用方保证。先 L2 normalize Q/K 并对 Q 缩放 `1/sqrt(Dk)`；然后计算衰减、预测、Delta 写入以及更新后的状态读出。返回新 state，原 state 不变。
- `delta_chunkwise` 输入 Q/K/V `[B,T,H,D]`、g/beta `[B,T,H]`、state `[B,H,Dk,Dv]`，支持 16/32/64 的 chunk；输出全序列与最终 FP32 state。每个 chunk 求单位下三角系统并在 chunk 间传递 state，最后不满 chunk 的长度无需填充。当前每个 Head 使用单线程顺序计算，是易验证的计算基线，不承诺并行性能。
- `delta_output(x,z,weight)` 沿最后一维执行 FP32 RMSNorm 后乘以 weight 和 SiLU(z)；weight shape `[Dv]`。此门控不同于 Dense Attention 的 sigmoid。

## 阶段五：DSA

- `lightning_indexer(q,k,weights,lengths,positions)` 接受预处理后的 Q `[B,T,Hi,Di]`、连续 Index Key `[B,L,Di]`、权重 `[B,T,Hi]`，返回 FP32 `[B,T,L]`。先计算 `sum_h weights[h]*relu(q[h]·k)`；有效长度之外及绝对位置之后输出 `-inf`。模型相关的投影、RoPE、归一化与缩放由上游处理。
- `topk_select(scores,k)` 接受 CUDA FP32 `[B,T,L]`，返回 int32 `[B,T,k]`；按分数降序、同分时逻辑位置升序，非有限值不选，不足时填 `-1`。这是逐行顺序选择的正确性实现，复杂度较高；未实现 CUB 分段排序或 FP8。
- `sparse_mla(...)` 使用 Top-K 的逻辑位置，复用 MLA 分页计算核心；不将 Indexer score 作为主 Attention softmax 分数。

## 元数据与状态责任

GPU 上的长度、位置、物理页 ID、slot 和逻辑索引不会被同步回读验证。调用方必须保证长度在张量或页表容量内，所有使用到的物理页和槽合法，写入槽唯一，所引用缓存已经写入；Sparse MLA 的有效索引必须不重复。不同请求与不同 Stream 的状态和 Cache 不应有写入冲突；必要的同步、页面分配、Prefix Sharing 与回收由调用方负责。空注意力集合输出零。
