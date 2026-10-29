# Dense GQA prefill

`gqa_prefill(q, k, v, q_lens, kv_lens, causal=True, scale=None)` is a forward-only CUDA operator. Q is contiguous `[B,Tq,Hq,D]`; K and V are contiguous `[B,Tk,Hkv,D]`, on the same device and with the same FP32/FP16/BF16 dtype. Hq must be divisible by Hkv and `1 <= D <= 256`. Lengths are CUDA int32 `[B]`, with `0 <= q_lens[b] <= Tq` and `0 <= kv_lens[b] <= Tk`; callers must ensure valid values. Output has Q's shape and dtype, with padded rows and rows attending to no keys set to zero.

For causal attention, query row `i` aligns with key position `kv_lens[b] - q_lens[b] + i`. Keys up to and including that position are visible. This is right-aligned even when query and key lengths differ. Non-causal rows see all valid keys. GQA uses `kv_head = q_head // (Hq/Hkv)`.

The current SIMT kernel processes 32-token KV tiles and uses FP32 online softmax and accumulators. It does not materialize an attention matrix, but is a correctness-oriented implementation, not a Tensor Core FlashAttention kernel. Values of the length tensors are not synchronously checked by the operator.


`backend="tensorcore"` uses Triton, with 16 query rows and 32 keys per tile and FP32 softmax state. QK and PV use tensor-core dot operations; the probability tile is cast to the input dtype before PV, so rounding differs from the SIMT FP32 accumulation. The backend requires a compatible Triton installation and FP16/BF16 inputs. On the tested SM120 device the generated PTX includes `mma.sync.aligned.m16n8k16` for both FP16 and BF16. Neither backend reads GPU metadata back to the host to check its value ranges.
