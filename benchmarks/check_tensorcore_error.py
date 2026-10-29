"""Measure maximum absolute error of experimental WMMA versus CUDA defaults."""
import json,math,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import qd_attention as q
results=[]
for seed in (101,102,103):
 torch.manual_seed(seed);B,T,H,D=1,64,2,33
 x=torch.randn(B,T,H,D,device='cuda');k=torch.randn_like(x);v=torch.randn(B,T,H,35,device='cuda');g=-torch.rand(B,T,H,device='cuda')*.1;beta=torch.rand(B,T,H,device='cuda');state=torch.randn(B,H,D,35,device='cuda')
 a=q.delta_chunkwise(x,k,v,g,beta,state,chunk_size=64);b=q.delta_chunkwise(x,k,v,g,beta,state,chunk_size=64,backend='wmma')
 results.append({'operator':'delta','seed':seed,'dtype':'fp32','output_max_abs':(a[0]-b[0]).abs().max().item(),'state_max_abs':(a[1]-b[1]).abs().max().item()})
 for dtype in (torch.float16,torch.bfloat16):
  B,T,H,Dn,Dr,Dc,Dv,L=1,32,2,16,7,33,16,64
  x=torch.randn(B,T,H,Dn,device='cuda',dtype=dtype);qr=torch.randn(B,T,H,Dr,device='cuda',dtype=dtype);c=torch.randn(B,L,Dc,device='cuda',dtype=dtype);rope=torch.randn(B,L,Dr,device='cuda',dtype=dtype);wk=torch.randn(H,Dn,Dc,device='cuda',dtype=dtype)/math.sqrt(Dn);wv=torch.randn(H,Dv,Dc,device='cuda',dtype=dtype)/math.sqrt(Dc);lens=torch.tensor([L],device='cuda',dtype=torch.int32);pos=torch.arange(L-T,L,device='cuda',dtype=torch.int32).reshape(1,T)
  a=q.mla_prefill(x,qr,c,rope,wk,wv,lens,pos);b=q.mla_prefill(x,qr,c,rope,wk,wv,lens,pos,backend='wmma')
  results.append({'operator':'mla_prefill','seed':seed,'dtype':str(dtype),'output_max_abs':(a.float()-b.float()).abs().max().item()})
pathlib.Path('results/tensorcore_followup_20261030/error.json').write_text(json.dumps(results,indent=2)+'\n')
