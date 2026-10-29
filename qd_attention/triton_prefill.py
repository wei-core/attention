"""Tiled tensor-core GQA prefill (optional Triton backend)."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _prefill(Q, K, V, QL, KL, O, TQ: tl.constexpr, TK: tl.constexpr,
             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
             CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    row0 = tl.program_id(0) * BM
    h = tl.program_id(1)
    b = tl.program_id(2)
    kh = h // (HQ // HK)
    qlen = tl.load(QL + b)
    klen = tl.load(KL + b)
    rows = row0 + tl.arange(0, BM)
    cols = tl.arange(0, BD)
    ns = tl.arange(0, BN)
    q = tl.load(Q + ((b*TQ + rows[:, None])*HQ + h)*D + cols[None, :],
                mask=(rows[:, None] < qlen) & (rows[:, None] < TQ) & (cols[None, :] < D), other=0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    l = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, BD), 0., tl.float32)
    for start in range(0, tl.cdiv(TK, BN)):
        tokens = start * BN + ns
        key = tl.load(K + ((b*TK + tokens[None, :])*HK + kh)*D + cols[:, None],
                      mask=(tokens[None, :] < klen) & (tokens[None, :] < TK) & (cols[:, None] < D), other=0)
        val = tl.load(V + ((b*TK + tokens[:, None])*HK + kh)*D + cols[None, :],
                      mask=(tokens[:, None] < klen) & (tokens[:, None] < TK) & (cols[None, :] < D), other=0)
        scores = tl.dot(q, key) * SCALE
        valid = (rows[:, None] < qlen) & (rows[:, None] < TQ) & (tokens[None, :] < klen) & (tokens[None, :] < TK)
        if CAUSAL:
            valid = valid & (tokens[None, :] <= (klen - qlen + rows[:, None]))
        scores = tl.where(valid, scores, -float('inf'))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float('inf'), 0., new_m)
        alpha = tl.where(l > 0, tl.exp(m - safe_m), 0.)
        p = tl.where(valid, tl.exp(scores - safe_m[:, None]), 0.)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), val)
        l = l * alpha + tl.sum(p, 1)
        m = new_m
    out = acc / tl.where(l > 0, l, 1.)[:, None]
    tl.store(O + ((b*TQ + rows[:, None])*HQ + h)*D + cols[None, :], out,
             mask=(rows[:, None] < TQ) & (cols[None, :] < D))


def tensorcore_prefill(q, k, v, q_lens, kv_lens, *, causal=True, scale=None):
    """FP16/BF16 tensor-core tiled forward with right-aligned causal semantics."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('q/k/v must be [B,T,H,D]')
    if not all(x.is_cuda and x.is_contiguous() and x.device == q.device for x in (q,k,v,q_lens,kv_lens)):
        raise ValueError('all tensors must be contiguous and on one CUDA device')
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('tensorcore backend requires matching fp16 or bf16')
    b, tq, hq, d = q.shape
    if k.shape[0] != b or k.shape[3] != d or not 1 <= d <= 256 or not k.shape[2] or not hq or hq % k.shape[2]:
        raise ValueError('invalid batch, head mapping or head_dim')
    if q_lens.shape != (b,) or kv_lens.shape != (b,) or q_lens.dtype != torch.int32 or kv_lens.dtype != torch.int32:
        raise ValueError('lengths must be int32 [B]')
    if scale is None: scale = 1.0 / math.sqrt(d)
    if not math.isfinite(scale) or scale <= 0: raise ValueError('scale must be positive and finite')
    out = torch.empty_like(q)
    if b and tq:
        _prefill[(triton.cdiv(tq, 16), hq, b)](q,k,v,q_lens,kv_lens,out,tq,k.shape[1],hq,k.shape[2],d,float(scale),causal,16,32,max(32,triton.next_power_of_2(d)),num_warps=4)
    return out
