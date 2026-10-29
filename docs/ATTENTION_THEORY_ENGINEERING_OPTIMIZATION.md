# Attention 算子：模型来源、理论、工程实现与优化路线

本文覆盖 `qd_attention` 在提交 `0096124` 的全部公开前向计算路径。重点是当前代码**实际实现了什么**、已有数据能说明什么，以及下一步应观察什么；优化项是建议，不表示已经实现。本库是独立算子，不代表已接入或逐项复现所列模型。输入的模型投影、权重、RoPE、Cache/状态生命周期和设备 metadata 值域由上游负责。

## 1. 算子清单和模型来源

| 算子 / 路径 | 类型 | 理论或模型关联 | 当前版本 |
| --- | --- | --- | --- |
| `gqa_decode` | Dense Attention | GQA 用较少 KV heads 服务多个 Q heads；例如 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 为 Q=16、KV=8；[GQA 原论文](https://arxiv.org/abs/2305.13245) | Paged、单 CTA/Head、逐 Token SIMT 基线 |
| `gqa_decode_split` CUDA / Triton | Dense Attention | 同一 GQA 数学，Split-KV 是并行执行策略，可服务 Qwen 等 GQA 解码 | CUDA / Triton 两种分段 + 独立 merge |
| `gqa_decode_auto` | 调度 | 同上 | 根据静态容量与 GPU SM 数的启发式，不是另一种 Attention |
| `gqa_prefill` CUDA / Tensor Core | Dense Attention | Transformer causal attention；分块与在线归一化借鉴 [FlashAttention](https://arxiv.org/abs/2205.14135) | 单行 SIMT / Triton Tile MMA |
| `kv_append` | Cache 辅助 | 分页 KV 写入；[PagedAttention](https://arxiv.org/abs/2309.06180) 的页表思想用于服务系统 | 一 CTA/序列，合并 K/V 写入 |
| `latent_kv_write`、`mla_query_absorb`、`mla_value_project`、`mla_latent_attention`、`mla_prefill`、`mla_decode` | MLA 与配套 | [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) 引入低秩 MLA；[DeepSeek-V3 推理代码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)包含 MLA | 连续 Direct Prefill、Paged Absorbed Attention 与投影的正确性基线 |
| `gated_attention`、`causal_conv1d` / `causal_conv1d_update`、`delta_recurrent`、`delta_chunkwise`、`delta_output` | Gated Attention / 线性递归及配套 | [Qwen3-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)列出 Gated Attention 和 Gated DeltaNet 的混合层；[Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464) | 对核心数学做 CUDA 正确性基线；不是完整 Qwen block |
| `lightning_indexer`、`topk_select`、`sparse_mla` | DSA 与配套 | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 引入 Lightning Indexer + token 稀疏选择 | FP32 Indexer 分数、确定性顺序 Top-K、复用分页 MLA |

MHA、MQA、GQA 是 Head 比例，不是当前代码中三套独立 kernel。在线 Softmax、warp/block 归约是内部组件；`Compressed`、Backward、FP8、分布式和 Nano-vLLM 模型集成目前未实现。完整调用契约见 [Dense](KERNEL_CONTRACT.md)、[Prefill](PREFILL_CONTRACT.md)、[阶段三至五](STAGE345_CONTRACT.md)。

## 2. 理论、测量和证据如何对应

### 数学与理论可能收益

- Dense：`O=softmax(QKᵀ/√D)V`。GQA 的 KV **存储量**相对相同 Head 维的 MHA，理想比例为 `Hkv/Hq`；例如 Q=32、KV=8 是 1/4。QK/PV 计算仍需为每个 Q head 计算，吞吐不会自动提高 4 倍。不同 Q head 是否复用 KV 取决于 kernel 和 cache。
- Paged KV：`physical=table[b,t/P]`，页内 slot 为 `t%P`。它可缓解容量碎片、支持页面管理，不保证单次 Attention 算子必然更快。FlashAttention 的核心是 Tile、片上在线 Softmax 和避免物化完整 score；标准 Dense Attention 的乘法计算量仍随 `T×L` 增长。
- MLA：`q̃=W_UKᵀq_nope`，`score=scale(q̃·c+q_rope·k_rope)`，`u=Σsoftmax(score)c`，`o=W_UV u`。吸收前后数学等价，低精度舍入可能不同。理论原始 KV 元素数从每 Token `H(Dn+Dr+Dv)` 变成 `Dc+Dr`，不包含页表、权重和额外数据；减少缓存容量不等于同倍数减少时延。
- DeltaNet：`S̄=exp(g)S`，`Δ=β(v-kᵀS̄)`，`S'=S̄+kΔᵀ`，`y=qᵀS'`；本实现先归一化 Q/K。递归每步状态规模与 `Dk×Dv` 相关，不扫描所有历史 KV；Chunkwise 用下三角求解组织 chunk 内计算、chunk 间传递状态，但并不自动代表当前串行实现具备并行加速。
- DSA：Indexer `I_s=Σ_h w_h ReLU(q_h·k_s)`，Top-K 决定位置集合，再用**主 MLA score**在集合内 Softmax。理想情况下主 Attention 的历史遍历从 L 个位置降到 K 个；Indexer、Top-K、投影和随机访存仍需计费，`K/L` 不是端到端加速率。

### 本机数据和性能变化的边界

所有下文数字来自 [全算子性能实测](ALL_ATTENTION_OPERATORS_PERFORMANCE.md)，RTX 5060、SM120、PyTorch 2.9.1+cu128，2026-10-29；固定形状，两次进程、各五组的中位数再取中位数。Dense 为 BF16；MLA/Qwen/DSA 为 FP32 小形状。CUDA Events 覆盖 Python 算子路径、分配和可能的主机提交空隙，并非纯 kernel 延迟。GPU 未隔离、时钟未锁定；微小差异不应当视为显著。本轮不做调优；同一版本的路径对照不是“优化前后提交对照”。历史两轮 Dense 源码未变时的波动见 [历史复测](ALL_ATTENTION_OPERATORS_PERFORMANCE.md#同条件历史复测变化不等于优化收益)。

官方数据不能套到本机：[DeepSeek FlashMLA 固定版本](https://github.com/deepseek-ai/FlashMLA/blob/1aef31d163f11c0f2c7329fcda71d2985bfb47a0/README.md)在 H800、CUDA 12.8 上给出的 3000 GB/s 和 580 TFLOPS 分别对应不同受限配置，且与本机 GPU、dtype 和形状不同；[Sparse FlashMLA 技术报告](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)的 410 TFLOPS 使用 H800、FP8 KV、B=128、H=128、Q=2、Top-K=2048。它们不是 RTX 5060 本算子的目标延迟。

### 可计算的例子与不能推断的结论

1. Dense Decode，B=1、L=4096、Hq=32、Hkv=8、D=128、BF16：有效 K/V 存储量 `2×4096×8×128×2=16,777,216 B`。按 [NVIDIA 规格](https://www.nvidia.com/en-au/geforce/10-series/10-series-specs/) RTX 5060 标称 **448 GB/s**，若假设恰好从 DRAM 读一次且达到标称带宽，则 `bytes/bandwidth≈37.4 µs`。这只是一种理想流量模型：未计写出、页表、Q、重复读取、计算、分配和启动；缓存命中时也不是严格下界。本机顺序 CUDA **2890.8 µs**、Triton Split-4 **116.8 µs**，但不能用 37.4/116.8 推断硬件利用率或单独定位瓶颈。
2. MLA 的模型维度示例 H=128、Dn=128、Dr=64、Dv=128、Dc=512、BF16：按完整每 Head K/V 展开计 81,920 B/Token；共享 latent + rope 为 1,152 B/Token，理论原始存储比约 **71.1:1**。这不是对当前代码实测的时延提升，也不是与一切共享 RoPE/压缩布局的公平对比。本机 FP32 小形状 H=4、Dn=16、Dr=8、Dv=16、Dc=32 的简单存储比则是 **4:1**。
3. 本机 B=1、L=4096 的 Triton Split-4 为 116.8 µs，相对同次顺序 CUDA 的 2890.8 µs 延迟下降约 **96.0%**。这包含不同分段、不同 launch 和 workspace 路径，仅是该固定形状的 operator 观察，不可推广到所有 Batch/长度；例如 B=1、L=128 时两者分别为 90.2 与 88.5 µs。

## 3. 哪些指标能证明瓶颈

当前已有 Nsight Systems 时间线：BF16 B=1、L=1024 的顺序 CUDA kernel 中位数 672.4 µs（7 次含预热），另一次 Triton trace 的 `_decode` 27.7 µs 和 `_merge` 0.75 µs（各 15 次含预热）；记录见 [CUDA 汇总](../results/stage1_20261029/nsys_cuda_summary.txt) 与 [Triton 汇总](../results/stage1_20261029/nsys_triton_summary.txt)。两份 trace 是独立测量，不能直接相加当作本轮同条件 operator 比值。

Nsight Compute 在普通用户和 sudo 下均返回 `ERR_NVGPUCTRPERM`，见 [失败记录](../results/stage1_20261029/ncu_permission.log)。因此以下是**需采集的指标，不是本机已观测的数值**。参考 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：

| 问题 | 首先看什么 | 如何决定动作 |
| --- | --- | --- |
| Launch / 分配是否占主导 | Nsight Systems 中 CPU 提交、GPU 空隙、kernel 数；普通 Event operator 与预分配/CUDA Graph kernel-only 分开测 | 短算子时间接近 launch 或 CPU 间隙时，优先复用 workspace、减少 Python/launch；Graph 数字单独报告。 |
| 小 Grid / 并发是否不足 | Launch Statistics 的 grid/CTA、SM 数、活跃 warp / eligible warp | B×H 很小时试 Split-KV 或合并多 head；再测 merge 与 scratch 成本，不把 occupancy 当最终目标。 |
| 是否确为带宽或访问问题 | Memory Workload Analysis 的 DRAM 字节/吞吐、L1/L2 命中、sectors/request / excessive sectors、读写事务 | 带宽高且接近可持续峰值时减少数据或重复读；事务浪费高时检查对齐/合并；cache 命中高时不以逻辑字节量冒充 DRAM 流量。 |
| 寄存器、共享内存、溢出 | Occupancy / Launch Statistics 的寄存器每线程、shared memory、理论与实际 occupancy；local-memory load/store、spill | 先确认限制资源，再调整 Tile/Warp、累加器或重算，避免只追求高 occupancy。 |
| Tensor Core 是否有效 | Compute Workload Analysis 的 Tensor pipe 活动、指令计数、利用率；PTX/SASS 的 MMA 指令 | PTX 有 `mma.sync` 只证明路径，不证明利用率；对照 shape、dtype、mask 和编译产物。 |
| 是否同步或依赖受限 | Warp State / Scheduler Statistics 的 eligible warps、barrier、long/short scoreboard 等 stall | 结合采样来源定位同步、依赖链与访存；比例本身不能证明优化方向。 |

## 4. Dense 与 KV 算子逐项

### `gqa_decode`：分页顺序基线

**原理与理论。** 每个 Query Head 按 `h//(Hq/Hkv)` 选择 KV Head，并顺序扫描有效 token，以 `(m,l,acc)` 在线 Softmax 累加，避免存储完整 score。

**代码版本。** [CUDA](../csrc/dense/gqa_decode.cu) 为 `grid=(Hq,B)`、每 CTA 一个 Head、32/64/128/256 线程；每 token 读取页表、逐元素加载 Q/K/V，QK 经 [warp shuffle + shared-memory block 归约](../csrc/common/block_reduce.cuh)，FP32 累加。Q 也在循环中加载；不同 Q heads 没有在同一 CTA 显式共享 KV。源码没有显式 `half2`/`float4` 向量加载，也没有 Tensor Core；线程沿 D 相邻访问是合并访存的布局意图，实际事务仍需测量。QK、Softmax 与 PV 在一个 kernel 内融合。

**初版与差距。** B=1/Hq=32/Hkv=8/D=128/P=16、BF16，L=128/1024/4096 分别 **88.5 / 672.4 / 2890.8 µs**。L 增长时耗时增加是观察；是否由同步、重复加载或带宽主导仍需 §3 的计数器证明。候选：让每 Head 多 token 并行、缓存 Q、复用共享 KV、对齐后试向量 load；分别观察 kernel 时间、实际 DRAM/事务、barrier 和寄存器，重新跑正确性。

### `gqa_decode_split`：CUDA 与 Triton

**原理。** 每段各自计算 `(m,l,a)`，合并时按全局最大值重新缩放指数和及加权向量；短序列额外 merge 可能超过并行收益。

**代码版本。** CUDA 在 [同一文件](../csrc/dense/gqa_decode.cu) 增加 grid.z 分段，并把 FP32 `(m,l,acc)` 写 scratch，由第二个 kernel merge；支持调用方预分配 `[B,Hq,S,D+2]`。每段内部仍逐 Token。Triton [实现](../qd_attention/triton_decode.py)在每程序中并行处理 **BN=32** 个 Token、沿 D 规约，`num_warps=4`，输出 FP32 partial，再启动 `_merge`；`tl.load` 是张量化表达和相邻 D 地址，不能仅据此声称生成了特定宽度向量指令。没有对多个 Q heads 显式共享 KV，也不是 Tensor Core MMA。

**初版与差距。** B=1、L=4096：CUDA 顺序 **2890.8**、CUDA Split-4 **749.2**、Triton Split-4 **116.8 µs**；对顺序基线分别 **-74.1% / -96.0%**。B=1、L=128 时 Triton Split-4 **90.2** 高于顺序 **88.5 µs**。实际最优 splits 会随 B、L、D、设备与 scratch 成本改变；对比多组分段和 Nsys 两次 launch；观察 global sectors、cache 命中、CTA 利用与 spill，再决定合并、分组复用或分段。

### `gqa_decode_auto`

[Python 调度](../qd_attention/dense.py) 在静态 `table.shape[1]×page_size <512` 时走 CUDA 基线，否则结合 `B×Hq`、SM 数与容量计算 splits（上限 16），走 Triton；不读取真实 GPU 长度，不是离线自动调优。B=1、L=4096 **213.5 µs**，该形状显式 Triton Split-4 **116.8 µs**；两者 workspace 分配和 splits 不同，不能只归因于选错 splits。建议记录实际选择、可复用 workspace，并在固定输入下分别测调度、分配和 kernel；实际长度远小于容量时额外验证。

### `gqa_prefill`：SIMT 与 Triton Tensor Core

**原理。** 多个 Query 逐行计算 causal attention；右下对齐 mask 与变长语义见 [Prefill 契约](PREFILL_CONTRACT.md)。

**代码版本。** [CUDA SIMT](../csrc/dense/gqa_prefill.cu) 每 CTA 一个 `(B,T,H)` 行，外层按 32 token 分组但内层**逐 token 顺序**做点积、block reduction 和 FP32 在线 Softmax；这不是多个 Query 的矩阵 Tile，也没有 shared-memory KV tile 或 Tensor Core。 [Triton 路径](../qd_attention/triton_prefill.py) 使用 `BM=16, BN=32, num_warps=4`，`tl.dot` 计算 QK 和 PV、FP32 状态，概率转换为输入 dtype 后做 PV；测试时生成的 PTX 有 FP16/BF16 `mma.sync`，尚无利用率计数器。该实现遍历完整 TK 的 tile 并应用 mask，未明确跳过全因果遮挡 tile。

**初版与差距。** BF16、B=1、Hq=8/Hkv=2、D=64、L=1024：SIMT **3687.3 µs**，Triton **117.0 µs（-96.8%）**，本机 PyTorch SDPA **62.0 µs**、FlashAttention 2.8.3 **110.6 µs**。这些是不同库 wrapper 的 operator 测量；FlashAttention 的同机实测不等于官方固定指标。试不同 BM/BN/Warp、跳过遮挡 tile、共享/重用 KV；先看 Tensor pipe 活动、Tensor 指令、spill、shared memory 与真实 kernel 时间，不能仅因 `mma.sync` 认定已达峰值。

### `kv_append`

[CUDA](../csrc/cache/kv_append.cu) 每序列一个 256 线程 CTA，解表后逐元素同时写 K 和 V，属于**两个缓存写入融合到一个 kernel**；源码无显式向量 load/store。每个 batch 的目标页已分配且不同写入不能冲突。本机 BF16、B=1、Hkv=8/D=128：**15.3 µs**；PyTorch 已预计算索引、两次赋值参考 **40.8 µs**，工作分解不同且短算子抖动明显。若 Launch 占主导考虑合批/与后续计算融合；若事务浪费高再验证对齐并比较 `half2`/更宽 load-store，检查写入竞争及 cache 生命周期。

## 5. MLA 与潜在缓存逐项

相关模型为 DeepSeek-V2/V3 的 MLA 数学；这不是将本代码声称为官方 FlashMLA。主结果为 FP32、B=1/H=4/L=64/Dn=16/Dr=8/Dc=32/Dv=16；对照 PyTorch 是连续数学参考，不代表优化库。实现见 [MLA 代码](../csrc/mla/mla.cu) 与 [Direct Prefill](../csrc/mla/mla_prefill.cu)。

| API | 原理及当前工程版本 | 本机初版与差距 | 待验证的优化、对应指标 |
| --- | --- | --- | --- |
| `latent_kv_write` | 输入 `[N,Dc]` 与 `[N,Dr]` 沿特征拼接到 **物理 slot**；一个 CTA/Token、256 线程、标量元素拷贝，两个输入在一次 kernel 写入 | N=64 **27.9 µs**，PyTorch concat+索引写 **61.3 µs**；后一项不是等价单 kernel 性能基准 | 关注 Launch、目标分散度与写入 sectors；按对齐条件试向量写/批处理，不能破坏唯一 slot 或模型 RoPE 约定。 |
| `mla_query_absorb` | `q̃_c=Σ_d q_d W_UK[d,c]`；每线程负责一个输出 c，沿 Dn 串行累加 FP32 | T=4 **27.0 µs**，PyTorch einsum **65.5 µs**；小形状多受调用与并行粒度影响，尚无 kernel-only 证据 | 比较 batched GEMM / Tensor Core 和多个 rows 同 tile；看 SM/Tensor pipe、每线程寄存器、访存重复及转换精度。 |
| `mla_value_project` | `o_v=Σ_c u_c W_UV[v,c]`；每线程输出一个 v，沿 Dc 串行，FP32 累加后写输入 dtype | T=4 **22.2 µs**，参考 **66.0 µs**；不要把两次包装差值当官方优化收益 | 同上，评估合批、矩阵布局与投影合并；看 kernel 时间及权重读取，验证低精度的累积误差。 |
| `mla_latent_attention` | Paged `q̃·c+q_rope·k_rope`，一 CTA/`(B,T,H)`、256 线程、block reduction、FP32 在线 Softmax；Dc≤512，每线程最多两个累加值 | T=4 **53.6 µs**；参考连续 KV **343.1 µs**，参考不含分页 Gather，不能作为同布局优化幅度 | 对长 L 研究 Split-KV 和多 Q/Head 共享 latent，观察小 grid、L2/DRAM bytes、归约 barrier；指标未采集。 |
| `mla_prefill` | Direct 连续形式；每 token 对各 Key 重新按 W_UK 恢复内容，对 Value 按 W_UV 重算，再在线归一化；单行 CTA，无 Tensor Core | T=4 **128.5 µs**，Direct PyTorch 参考 **722.0 µs**；主要形状很小，无法外推真实模型 | 先量投影重复次数和 kernel 时间；再对照物化/分块 GEMM 或 Absorbed 数学，验证绝对位置与精度；看 Tensor pipe、SM 与访存。 |
| `mla_decode` | Python 依次启动 query absorption、paged latent attention 和 value projection；中间输出按输入 dtype 保存 | T=1 **67.2 µs**；连续 Direct 参考 **734.0 µs**，布局与求值顺序不同 | Nsight Systems 先看三个 launch、临时 Tensor；分离各阶段计时后评估 workspace 或融合，不能假定融合一定更快。 |

## 6. Qwen Hybrid 逐项

Qwen3-Next 的模型卡说明混合层；本库只实现对应数学核心，未运行真实权重和模型前后处理。FP32 测试中 Qwen 的 B=2、T=16（递归单步 T=1），完整输入与参考见 [原始数据](../results/all_operators_20261029/stage345_single_run1.json)。

| API | 原理与具体实现 | 当前测量 / 不能推断什么 | 候选优化与验证指标 |
| --- | --- | --- | --- |
| `gated_attention` | `attn×sigmoid(raw_gate)`，一线程一元素、一个 256 线程 CTA 覆盖多个元素；sigmoid 与乘法在同一 kernel，标量读写 | H=4/Dv=32 **22.4 µs**；逐元素 PyTorch 参考 **40.6 µs** | 看 launch 空隙、内存事务和指数指令；若调用占主导，可考虑与 Attention 输出融合，需考虑寄存器和 gate 语义。 |
| `causal_conv1d` | `[B,T,C]` depthwise 窗口点积 + SiLU + 生成新 `[B,C,K]` state；每线程一个 channel，时间 T 与 K 顺序循环，卷积、激活和 state 输出融合 | C=64/K=4/T=16 **24.7 µs**；逐 Token PyTorch 循环 **1799.1 µs**，参考产生许多 launch，不能当成熟卷积库的差距 | 按时间并行或 Tile 复用窗口，考察 input/state 访问事务、Launch 和寄存器；严格保证权重先旧后新和 state 的不变性。 |
| `causal_conv1d_update` | Python 用长度为 1 的输入调用同一卷积 kernel，返回新 state，不原地覆盖 | T=1 **43.8 µs**；Python 一步参考 **196.9 µs**；小形状在两轮中波动 | 先分离 wrapper、slice 与 kernel；看 Nsys 时间线，候选专用单步路径/状态缓冲复用，不要以一个短样本推断。 |
| `delta_recurrent` | 一 CTA/Head，线程映射 Dv，各线程串行遍历 Dk；Q/K L2 归一化、exp(g) 衰减、预测、Delta 写和读出在单 kernel；state/g/beta FP32；每个 Dv 线程重复算 Q/K 范数 | H=4/Dk=Dv=16 **25.1 µs**；PyTorch 多操作单步 **508.5 µs**，不是官方 fused kernel | 观察 Q/K 重复读取、线程活跃率、state 事务和小 grid；可先共享归一化、分块 Dk/Dv 或合并多 head，再对照更新后状态。 |
| `delta_chunkwise` | 每 Head **一个 CTA、仅一个线程**，FP32 global scratch，显式前向代入单位下三角系统、依次传递 chunk state；不是并行矩阵版本 | H=4/T=16/Dk=Dv=16/chunk=16 **750.2 µs**；PyTorch 16 步参考 **8518.4 µs**，其 Python 循环非常慢，并不表明此版本接近并行理论 | Launch Statistics 可先确认 CTA/thread 配置；候选把列/矩阵计算分配多线程、分离三角求解与矩阵运算，测活跃线程、访存和寄存器，并对 16/32/64、不满尾块和 state 精度回归。 |
| `delta_output` | 每行 256 线程，block 归约 `mean(x²)`，融合 RMSNorm、weight 和 SiLU(z)；不同于 sigmoid gate | H=4/Dv=32/T=16 **33.3 µs**，PyTorch 多操作参考 **148.3 µs**；D=32 时多数线程无有效特征 | 看活跃线程、block 规模、barrier 和 Launch；可比较 warp-per-row、多行 CTA 或与递归输出融合，验证 FP32 归约与数值。 |

## 7. DSA 与 Sparse MLA 逐项

此处 `index_q`、`index_k`、head weights 已完成模型要求的投影、位置编码、归一化与缩放；当前不实现官方 FP8 Index Cache 或正交变换。Sparse MLA 使用单独的 MLA 分数，不使用 Indexer 分数作为注意力权重。

| API | 原理与具体实现 | 当前测量 / 局限 | 候选优化与验证指标 |
| --- | --- | --- | --- |
| `lightning_indexer` | [CUDA](../csrc/sparse/indexer.cu) 每线程一个 `(B,T,L)` score，按 Head、D **串行循环**，FP32 `Σw·ReLU(dot)`；future / 越界 `-inf`；Index Key 为连续布局 | B=1/T=4/Hi=4/Di=16/L=64 **26.9 µs**；PyTorch einsum 参考 **348.4 µs**，小尺寸、多操作参考不能代表官方 | 对长 L/Hi/Di 验证，研究并行 dot、Q/K tile 复用和可选量化；观察活跃 warp、L2/DRAM 事务、寄存器和输出流量。 |
| `topk_select` | 每线程处理一行，初始化 byte `used[L]`，每轮完整扫描 L 找最大，共 K 轮；同分取较小逻辑索引，非有限值不选，不足填 -1；**不是 CUB** | B=1/T=4/L=64/K=16 **25.2 µs**；PyTorch stable argsort 参考 **482.7 µs**；对 L=64、K=16 的结果不能外推 K=2048 | 分段选择 / CUB / warp reduction，保持 tie、NaN、空行契约；观测 K/L 扩大后的耗时、活跃线程、used-buffer 读写与额外 workspace。 |
| `sparse_mla` | Python 先 absorb，再复用 MLA paged kernel 按选中**逻辑位置**读取，最后 value projection；跳过 `-1` 与无效位置 | T=1/L=64/K=16 **59.4 µs**；PyTorch 参考 **400.4 µs**，参考预吸收 Q 且算全 L 的 masked score，工作不同；本次不含 Indexer 和 Top-K | 端到端分别计 Indexer、Top-K、分页读取、投影与 launch；看随机 page sectors、cache hit、K/L 和 CTA 数；考虑重排/按页分组时保持逻辑顺序和重复索引语义。 |

## 8. 分阶段行动和验收

1. **建立可解释的现状**：在空闲设备固定版本、时钟/功耗策略和输入，重复不同进程，分别保存 operator、预分配工作区和 CUDA Graph kernel-only；不得把不同语义/布局混入同一“加速”列。
2. **补硬件证据**：先按 [NVIDIA 说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)解决宿主机性能计数器权限，再对代表性短/长、B=1/4/16 收集 §3 的指标和 Nsys 时间线；每条优化建议写明“观察 → 假设 → 实验 → 回归”。
3. **单独改变一个因素**：先研究 Decode 分段和 workspace、再研究 Prefill Tile 与真实 MMA 活动；MLA 投影和分页读取分别测量；Qwen Chunkwise/DSA Top-K 先改并行组织，同时保留当前数学基线。
4. **验收**：固定模型语义、位置、尺度和预处理；对 FP32/FP16/BF16、空/变长/分页/状态边界回归，再进行多形状性能比较。真实 DeepSeek/Qwen 权重、端到端模型输出、生产负载、Tensor Core 利用率和 CUB Top-K 均尚未验证或实现。

全部两轮样本、对照和复现命令见 [全算子性能实测](ALL_ATTENTION_OPERATORS_PERFORMANCE.md)；其模型、理论和官方指标的可比范围比单独引用某个峰值更重要。
