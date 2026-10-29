"""Direct and absorbed MLA references; inputs are normalized/rotated upstream."""
import math
import torch


def mla_reference(q_nope,q_rope,latent,rope_key,w_uk,w_uv,positions,lengths,*,scale=None,absorbed=False,indices=None):
    bsz,tq,heads,dn=q_nope.shape
    length=latent.shape[1]
    if scale is None: scale=1/math.sqrt(dn+q_rope.shape[-1])
    q=q_nope.float(); qr=q_rope.float(); c=latent.float(); kr=rope_key.float(); wk=w_uk.float(); wv=w_uv.float()
    if absorbed:
        qa=torch.einsum('bthd,hdc->bthc',q,wk)
        scores=torch.einsum('bthc,bsc->bths',qa,c)
    else:
        keys=torch.einsum('bsc,hdc->bshd',c,wk)
        scores=torch.einsum('bthd,bshd->bths',q,keys)
    scores=(scores+torch.einsum('bthr,bsr->bths',qr,kr))*scale
    ids=torch.arange(length,device=q.device).view(1,1,1,length)
    mask=(ids<lengths[:,None,None,None]) & (ids<=positions[:,:,None,None])
    if indices is not None:
        chosen=torch.zeros((bsz,tq,length),device=q.device,dtype=torch.bool)
        for b in range(bsz):
            for t in range(tq):
                ix=indices[b,t]; ix=ix[(ix>=0)&(ix<length)].long(); chosen[b,t,ix]=True
        mask=mask & chosen[:,:,None,:]
    scores=scores.masked_fill(~mask,-torch.inf)
    any_valid=mask.any(dim=-1,keepdim=True)
    weights=torch.softmax(torch.where(any_valid,scores,torch.zeros_like(scores)),dim=-1)
    weights=torch.where(any_valid,weights,torch.zeros_like(weights))
    if absorbed:
        u=torch.einsum('bths,bsc->bthc',weights,c)
        return torch.einsum('bthc,hvc->bthv',u,wv).to(q_nope.dtype)
    values=torch.einsum('bsc,hvc->bshv',c,wv)
    return torch.einsum('bths,bshv->bthv',weights,values).to(q_nope.dtype)
