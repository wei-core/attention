"""DSA Lightning Indexer and deterministic top-K math references."""
import torch
import torch.nn.functional as F


def lightning_indexer_reference(q,k,weights,lengths,positions):
    dots=torch.einsum('bthd,bsd->bths',q.float(),k.float())
    scores=(F.relu(dots)*weights.float().unsqueeze(-1)).sum(2)
    ids=torch.arange(k.shape[1],device=k.device)[None,None,:]
    valid=(ids<lengths[:,None,None]) & (ids<=positions[:,:,None])
    return scores.masked_fill(~valid,-torch.inf)


def topk_reference(scores,k):
    scores=scores.masked_fill(~torch.isfinite(scores),-torch.inf)
    order=torch.argsort(scores,dim=-1,descending=True,stable=True)
    n=min(k,scores.shape[-1]); out=torch.full((*scores.shape[:-1],k),-1,dtype=torch.int32,device=scores.device)
    ix=order[...,:n]; vals=torch.gather(scores,-1,ix)
    out[...,:n]=torch.where(torch.isfinite(vals),ix,-1).int()
    return out
