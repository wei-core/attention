"""Forward-only Qwen attention gate and Gated DeltaNet primitives."""
import torch
from .dense import is_cuda_extension_loaded


def _ready():
    if not is_cuda_extension_loaded(): raise RuntimeError('CUDA extension unavailable')


def gated_attention(attn, gate):
    """Apply sigmoid to raw per-feature gate logits after attention."""
    _ready(); return torch.ops.qd_attention.gated_attention(attn,gate)


def delta_recurrent(q,k,v,g,beta,state,*,eps=1e-6):
    """One step: normalized Q/K, FP32 log decay, beta and [B,H,Dk,Dv] state.

    Returns (output, new_state); does not mutate state. g is log decay, beta is
    already sigmoid-activated. No implicit state/cache lifecycle management.
    """
    _ready(); return torch.ops.qd_attention.delta_recurrent(q,k,v,g,beta,state,float(eps))


def delta_output(x,z,weight,*,eps=1e-6):
    """RMSNorm(x)*SiLU(z), with a per-feature norm weight."""
    _ready(); return torch.ops.qd_attention.delta_output(x,z,weight,float(eps))


def causal_conv1d(x, weight, state):
    """Depthwise causal convolution + SiLU, returning output and new state.

    x [B,T,C], weight [C,K], state [B,C,K] ordered oldest to newest.
    The incoming state is not modified; a one-token input is the decode update.
    """
    _ready(); return torch.ops.qd_attention.causal_conv1d(x,weight,state)


def delta_chunkwise(q,k,v,g,beta,state,*,chunk_size=32,eps=1e-6,backend="cuda"):
    """Chunkwise triangular solve, FP32 state; returns output and new state.

    ``backend="wmma"`` is experimental: it uses compensated TF32 WMMA for
    off-diagonal updates and requires chunk >= 32 and Dk+Dv >= 32 on SM80+.
    """
    if backend not in ("cuda", "wmma"): raise ValueError("backend must be cuda or wmma")
    _ready(); return torch.ops.qd_attention.delta_chunkwise(q,k,v,g,beta,state,chunk_size,float(eps),backend=="wmma")


def causal_conv1d_update(x, weight, state):
    """One-token decode update: x [B,C]; returns y [B,C] and new state."""
    if x.ndim != 2: raise ValueError('x must have shape [B,C]')
    y,new_state=causal_conv1d(x.unsqueeze(1),weight,state)
    return y[:,0,:],new_state
