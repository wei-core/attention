"""Paged GQA split decode with token-parallel, coalesced Triton loads."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, TABLE, LENS, PART, HQ:tl.constexpr, HK:tl.constexpr,
            D:tl.constexpr, P:tl.constexpr, MAXP:tl.constexpr, SPLITS:tl.constexpr,
            SCALE:tl.constexpr, BN:tl.constexpr, BD:tl.constexpr):
    h, b, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (HQ // HK)
    length = tl.load(LENS+b)
    begin = (length*s)//SPLITS
    end = (length*(s+1))//SPLITS
    ds = tl.arange(0, BD)
    q = tl.load(Q+(b*HQ+h)*D+ds, mask=ds<D, other=0).to(tl.float32)
    m = tl.full((), -float('inf'), tl.float32)
    l = tl.full((), 0., tl.float32)
    acc = tl.full((BD,), 0., tl.float32)
    for base in range(begin, end, BN):
        t = base + tl.arange(0, BN)
        valid = t < end
        page = tl.load(TABLE+b*MAXP+t//P, mask=valid, other=0)
        offsets = ((page[:,None]*P+t[:,None]%P)*HK+kh)*D+ds[None,:]
        k = tl.load(K+offsets, mask=valid[:,None] & (ds[None,:]<D), other=0).to(tl.float32)
        v = tl.load(V+offsets, mask=valid[:,None] & (ds[None,:]<D), other=0).to(tl.float32)
        score = tl.sum(k*q[None,:], 1)*SCALE
        score = tl.where(valid, score, -float('inf'))
        new_m = tl.maximum(m, tl.max(score, 0))
        a = tl.where(l > 0, tl.exp(m-new_m), 0.)
        weight = tl.where(valid, tl.exp(score-new_m), 0.)
        acc = acc*a + tl.sum(v*weight[:,None], 0)
        l = l*a+tl.sum(weight,0)
        m = new_m
    offset = ((b*HQ+h)*SPLITS+s)*(D+2)
    tl.store(PART+offset, m)
    tl.store(PART+offset+1, l)
    tl.store(PART+offset+2+ds, acc, mask=ds<D)


@triton.jit
def _merge(PART, OUT, HQ:tl.constexpr, D:tl.constexpr, SPLITS:tl.constexpr,
           BS:tl.constexpr, BD:tl.constexpr):
    h,b = tl.program_id(0),tl.program_id(1)
    ss,ds = tl.arange(0,BS),tl.arange(0,BD)
    base = ((b*HQ+h)*SPLITS+ss)*(D+2)
    m = tl.load(PART+base, mask=ss<SPLITS, other=-float('inf'))
    l = tl.load(PART+base+1, mask=ss<SPLITS, other=0.)
    maxm = tl.max(m,0)
    maxm = tl.where(maxm == -float('inf'), 0., maxm)
    weights = tl.where(l>0, tl.exp(m-maxm), 0.)
    den = tl.sum(weights*l,0)
    acc = tl.load(PART+base[:,None]+2+ds[None,:], mask=(ss[:,None]<SPLITS)&(ds[None,:]<D), other=0.)
    result = tl.sum(acc*weights[:,None],0)/tl.where(den>0,den,1.)
    tl.store(OUT+(b*HQ+h)*D+ds,result,mask=ds<D)


def paged_decode(q,k,v,table,lengths,*,splits=4,scale=None,workspace=None):
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape: raise ValueError('invalid q or cache shape')
    if not all(x.is_cuda and x.is_contiguous() and x.device==q.device for x in (q,k,v,table,lengths)): raise ValueError('all tensors must be contiguous on one CUDA device')
    b,h,d = q.shape
    if q.dtype not in (torch.float32,torch.float16,torch.bfloat16) or q.dtype != k.dtype or q.dtype != v.dtype: raise ValueError('invalid dtype')
    if not 1 <= d <= 256 or k.shape[3] != d or not h or not k.shape[2] or h % k.shape[2] or not k.shape[0] or not k.shape[1]: raise ValueError('invalid heads or cache')
    if table.ndim != 2 or table.shape[0] != b or not table.shape[1] or lengths.shape != (b,) or table.dtype != torch.int32 or lengths.dtype != torch.int32: raise ValueError('invalid metadata')
    if not isinstance(splits,int) or not 1 <= splits <= 64: raise ValueError('splits must be in [1,64]')
    if scale is None: scale = 1/math.sqrt(d)
    if not math.isfinite(scale) or scale <= 0: raise ValueError('scale must be positive and finite')
    shape = (b,h,splits,d+2)
    if workspace is None: workspace = torch.empty(shape,device=q.device,dtype=torch.float32)
    if workspace.shape != shape or workspace.dtype != torch.float32 or workspace.device != q.device or not workspace.is_contiguous(): raise ValueError('workspace must be float32 [B,Hq,splits,D+2]')
    out = torch.empty_like(q)
    if b:
        bd=max(32,triton.next_power_of_2(d))
        _decode[(h,b,splits)](q,k,v,table,lengths,workspace,h,k.shape[2],d,k.shape[1],table.shape[1],splits,float(scale),32,bd,num_warps=4)
        _merge[(h,b)](workspace,out,h,d,splits,triton.next_power_of_2(splits),bd,num_warps=4)
    return out
