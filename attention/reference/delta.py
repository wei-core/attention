"""Explicit FP32 Qwen-style DeltaNet and gate references."""
import math
import torch
import torch.nn.functional as F


def delta_recurrent_reference(q,k,v,g,beta,state,*,eps=1e-6):
    output_dtype=v.dtype
    q=q.float();k=k.float();v=v.float()
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+eps)/math.sqrt(q.shape[-1])
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+eps)
    old=state.float()*g.float().exp()[...,None,None]
    predicted=torch.einsum('bhd,bhdv->bhv',k,old)
    diff=beta.float()[...,None]*(v-predicted)
    new=old+k[...,None]*diff[...,None,:]
    out=torch.einsum('bhd,bhdv->bhv',q,new)
    return out.to(output_dtype),new


def delta_sequence_reference(q,k,v,g,beta,state,*,eps=1e-6):
    out=[]
    for t in range(q.shape[1]):
        y,state=delta_recurrent_reference(q[:,t],k[:,t],v[:,t],g[:,t],beta[:,t],state,eps=eps)
        out.append(y)
    return (torch.stack(out,1) if out else v.new_empty(v.shape)),state


def gated_attention_reference(attn,gate): return (attn.float()*torch.sigmoid(gate.float())).to(attn.dtype)

def delta_output_reference(x,z,weight,*,eps=1e-6):
    y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*weight.float()
    return (y*F.silu(z.float())).to(x.dtype)


def causal_conv1d_reference(x,weight,state):
    state=state.clone(); outputs=[]
    for t in range(x.shape[1]):
        state=torch.cat((state[:,:,1:],x[:,t,:,None]),-1)
        y=(state.float()*weight.float()[None,:,:]).sum(-1)
        outputs.append(F.silu(y).to(x.dtype))
    return (torch.stack(outputs,1) if outputs else x.new_empty(x.shape)),state
