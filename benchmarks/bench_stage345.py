"""Stage 3-5 operator latency and mathematical PyTorch reference comparisons."""
import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import qd_attention as qda
from qd_attention.reference.mla import mla_reference
from qd_attention.reference.delta import (gated_attention_reference,delta_output_reference,delta_recurrent_reference,delta_sequence_reference,causal_conv1d_reference)
from qd_attention.reference.sparse import lightning_indexer_reference,topk_reference


def cmd(*args):
    try: return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    except (OSError,subprocess.CalledProcessError) as e:return str(e)


def measure(fn,warmup,samples,inner):
    for _ in range(warmup):fn()
    torch.cuda.synchronize(); times=[]
    for _ in range(samples):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(inner):fn()
        b.record();b.synchronize();times.append(a.elapsed_time(b)*1000/inner)
    return {'median_us':statistics.median(times),'min_us':min(times),'max_us':max(times),'samples_us':times}


def check(actual,expected,tol):
    if isinstance(actual,(tuple,list)):
        for a,b in zip(actual,expected):check(a,b,tol)
    else: torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--warmup',type=int,default=5);p.add_argument('--samples',type=int,default=5);p.add_argument('--inner',type=int,default=5);args=p.parse_args()
    if not torch.cuda.is_available() or not qda.is_cuda_extension_loaded():raise RuntimeError('CUDA extension and GPU required')
    torch.manual_seed(620); rows=[]; dev='cuda'; dtype=torch.float32
    def bench(name,fn,ref,shape,note='',tol=1e-4):
        expected=ref();actual=fn();check(actual,expected,tol)
        r={'operator':name,'shape':shape,'dtype':'fp32','reference':note,'cuda':measure(fn,args.warmup,args.samples,args.inner),'pytorch_reference':measure(ref,args.warmup,args.samples,args.inner)}
        rows.append(r);print(json.dumps(r),flush=True)
    B,T,H,Dn,Dr,Dc,Dv,L,P=1,4,4,16,8,32,16,64,16
    q=torch.randn(B,T,H,Dn,device=dev);qr=torch.randn(B,T,H,Dr,device=dev);c=torch.randn(B,L,Dc,device=dev);kr=torch.randn(B,L,Dr,device=dev)
    wk=torch.randn(H,Dn,Dc,device=dev)/math.sqrt(Dn);wv=torch.randn(H,Dv,Dc,device=dev)/math.sqrt(Dc)
    lens=torch.tensor([L],device=dev,dtype=torch.int32);positions=torch.tensor([[60,61,62,63]],device=dev,dtype=torch.int32)
    table=torch.tensor([[2,0,3,1]],device=dev,dtype=torch.int32);cache=torch.empty(4,P,Dc+Dr,device=dev)
    slots=torch.tensor([int(x)*P+i for x in [2,0,3,1] for i in range(P)],device=dev,dtype=torch.int64)
    qda.latent_kv_write(c.reshape(-1,Dc),kr.reshape(-1,Dr),cache,slots)
    def write_ref(): cache.view(-1,Dc+Dr)[slots]=torch.cat((c.reshape(-1,Dc),kr.reshape(-1,Dr)),1)
    def write_cuda(): qda.latent_kv_write(c.reshape(-1,Dc),kr.reshape(-1,Dr),cache,slots)
    write_cuda(); actual=cache.view(-1,Dc+Dr).index_select(0,slots).clone();write_ref();expected=cache.view(-1,Dc+Dr).index_select(0,slots);check(actual,expected,0)
    # Both paths mutate the same cache with the same deterministic payload.
    bench('latent_kv_write',lambda:(write_cuda(),cache)[1],lambda:(write_ref(),cache)[1],{'tokens':L,'dc':Dc,'dr':Dr,'page':P},'PyTorch concat + indexed assignment',tol=0)
    bench('mla_query_absorb',lambda:qda.mla_query_absorb(q,wk),lambda:torch.einsum('bthd,hdc->bthc',q,wk),{'B':B,'T':T,'H':H,'Dn':Dn,'Dc':Dc},'PyTorch einsum')
    u=torch.randn(B,T,H,Dc,device=dev)
    bench('mla_value_project',lambda:qda.mla_value_project(u,wv),lambda:torch.einsum('bthc,hvc->bthv',u,wv),{'B':B,'T':T,'H':H,'Dc':Dc,'Dv':Dv},'PyTorch einsum')
    qa=qda.mla_query_absorb(q,wk);scale=1/math.sqrt(Dn+Dr)
    mask=torch.arange(L,device=dev)[None,None,None,:] <= positions[:,:,None,None]
    ix=torch.arange(0,32,2,device=dev,dtype=torch.int32).view(1,1,16).expand(B,T,16).contiguous()
    sparse_mask=torch.zeros(B,T,L,device=dev,dtype=torch.bool).scatter_(2,ix.long(),True)[:,:,None,:] & mask
    def latent_ref(sparse=False):
        s=(torch.einsum('bthd,bld->bthl',qa.float(),c.float())+torch.einsum('bthr,blr->bthl',qr.float(),kr.float()))*scale
        s=s.masked_fill(~(sparse_mask if sparse else mask),-torch.inf)
        return torch.einsum('bthl,bld->bthd',torch.softmax(s,-1),c.float()).to(qa.dtype)
    bench('mla_latent_attention',lambda:qda.mla_latent_attention(qa,qr,cache,table,lens,positions,scale=scale),latent_ref,{'B':B,'T':T,'H':H,'L':L,'Dc':Dc,'Dr':Dr},'Contiguous PyTorch dense einsums; excludes paged gather',tol=2e-4)
    direct=lambda:mla_reference(q,qr,c,kr,wk,wv,positions,lens)
    bench('mla_prefill',lambda:qda.mla_prefill(q,qr,c,kr,wk,wv,lens,positions),direct,{'B':B,'T':T,'H':H,'L':L,'Dn':Dn,'Dr':Dr,'Dc':Dc,'Dv':Dv},'Direct PyTorch contiguous math',tol=5e-4)
    q1=q[:,-1:].contiguous();qr1=qr[:,-1:].contiguous();pos1=positions[:,-1:].contiguous();ix1=ix[:,-1:].contiguous()
    qa1=qda.mla_query_absorb(q1,wk)
    sparse_mask1=sparse_mask[:,-1:].contiguous()
    def direct1():return mla_reference(q1,qr1,c,kr,wk,wv,pos1,lens)
    bench('mla_decode',lambda:qda.mla_decode(q1,qr1,cache,table,lens,pos1,wk,wv),direct1,{'B':B,'T':1,'H':H,'L':L,'Dn':Dn,'Dr':Dr,'Dc':Dc,'Dv':Dv},'Direct PyTorch contiguous math, different evaluation order',tol=5e-4)
    def sparse_ref():
        score=(torch.einsum('bthd,bld->bthl',qa1.float(),c.float())+torch.einsum('bthr,blr->bthl',qr1.float(),kr.float()))*scale
        u=torch.einsum('bthl,bld->bthd',torch.softmax(score.masked_fill(~sparse_mask1,-torch.inf),-1),c.float())
        return torch.einsum('bthc,hvc->bthv',u,wv)
    bench('sparse_mla',lambda:qda.sparse_mla(q1,qr1,cache,table,lens,pos1,wk,wv,ix1),sparse_ref,{'B':B,'T':1,'H':H,'L':L,'K':16,'Dc':Dc},'PyTorch masked dense scores with preabsorbed query and contiguous KV; different work',tol=5e-4)
    a=torch.randn(2,16,4,32,device=dev);z=torch.randn_like(a);weight=torch.randn(32,device=dev)
    bench('gated_attention',lambda:qda.gated_attention(a,z),lambda:gated_attention_reference(a,z),{'B':2,'T':16,'H':4,'Dv':32},'PyTorch elementwise sigmoid')
    bench('delta_output',lambda:qda.delta_output(a,z,weight),lambda:delta_output_reference(a,z,weight),{'B':2,'T':16,'H':4,'Dv':32},'PyTorch RMSNorm + SiLU',tol=2e-4)
    x=torch.randn(2,16,64,device=dev);cw=torch.randn(64,4,device=dev);cs=torch.randn(2,64,4,device=dev)
    bench('causal_conv1d',lambda:qda.causal_conv1d(x,cw,cs),lambda:causal_conv1d_reference(x,cw,cs),{'B':2,'T':16,'C':64,'K':4},'PyTorch per-token loop',tol=2e-4)
    xt=x[:,0].contiguous()
    def update_ref():
        y,s=causal_conv1d_reference(xt.unsqueeze(1),cw,cs);return y[:,0],s
    bench('causal_conv1d_update',lambda:qda.causal_conv1d_update(xt,cw,cs),update_ref,{'B':2,'T':1,'C':64,'K':4},'PyTorch one-step update',tol=2e-4)
    b,t,h,dk,dv=2,16,4,16,16
    qq=torch.randn(b,t,h,dk,device=dev);kk=torch.randn_like(qq);vv=torch.randn(b,t,h,dv,device=dev);g=-torch.rand(b,t,h,device=dev)*.2;beta=torch.rand(b,t,h,device=dev);state=torch.randn(b,h,dk,dv,device=dev)
    q1,k1,v1,g1,b1=(x[:,0].contiguous() for x in (qq,kk,vv,g,beta))
    bench('delta_recurrent',lambda:qda.delta_recurrent(q1,k1,v1,g1,b1,state),lambda:delta_recurrent_reference(q1,k1,v1,g1,b1,state),{'B':b,'T':1,'H':h,'Dk':dk,'Dv':dv},'PyTorch recurrent step',tol=2e-4)
    bench('delta_chunkwise',lambda:qda.delta_chunkwise(qq,kk,vv,g,beta,state,chunk_size=16),lambda:delta_sequence_reference(qq,kk,vv,g,beta,state),{'B':b,'T':t,'H':h,'Dk':dk,'Dv':dv,'chunk':16},'PyTorch sequential recurrence; algorithm differs',tol=2e-4)
    iq=torch.randn(1,4,4,16,device=dev);ik=torch.randn(1,64,16,device=dev);iw=torch.randn(1,4,4,device=dev)
    bench('lightning_indexer',lambda:qda.lightning_indexer(iq,ik,iw,lens,positions),lambda:lightning_indexer_reference(iq,ik,iw,lens,positions),{'B':1,'T':4,'Hi':4,'Di':16,'L':64},'PyTorch einsum + relu + masked sum',tol=2e-4)
    scores=qda.lightning_indexer(iq,ik,iw,lens,positions)
    bench('topk_select',lambda:qda.topk_select(scores,16),lambda:topk_reference(scores,16),{'B':1,'T':4,'L':64,'K':16},'PyTorch stable argsort; nonfinite excluded',tol=0)
    meta={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'git_head':cmd('git','rev-parse','HEAD'),'git_status':cmd('git','status','--short'),'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'python':platform.python_version(),'gpu_info':cmd('nvidia-smi','--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,power.draw,memory.used','--format=csv'),'method':'CUDA events on repeated Python calls; includes allocation, submission gaps and reference decomposition','warmup':args.warmup,'samples':args.samples,'inner':args.inner,'seed':620}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({'environment':meta,'results':rows},indent=2)+'\n')

if __name__=='__main__': main()
