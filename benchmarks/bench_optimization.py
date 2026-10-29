"""Operator comparison across separately built checkouts; CUDA-event timings in microseconds."""
import argparse, json, math, pathlib, statistics, sys, datetime
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=7);p.add_argument('--inner',type=int,default=10);args=p.parse_args()
sys.path.insert(0,str(pathlib.Path(args.root).resolve()))
import torch
import qd_attention as q

torch.manual_seed(936);dev='cuda';rows=[]
def bench(name,fn,shape):
    for _ in range(5): fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(args.repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(args.inner):fn()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/args.inner)
    rows.append(dict(name=name,shape=shape,median_us=statistics.median(samples),samples_us=samples))
    print(name,shape,round(rows[-1]['median_us'],3),flush=True)
for L in (128,512,1024):
    dtype=torch.bfloat16; h,hk,d=8,2,64
    qv=torch.randn(1,L,h,d,device=dev,dtype=dtype);kv=torch.randn(1,L,hk,d,device=dev,dtype=dtype);vv=torch.randn_like(kv);lens=torch.tensor([L],device=dev,dtype=torch.int32)
    bench('gqa_prefill_cuda',lambda:q.gqa_prefill(qv,kv,vv,lens,lens),{'L':L,'H':h,'Hkv':hk,'D':d,'dtype':'bf16'})
    bench('gqa_prefill_triton',lambda:q.gqa_prefill(qv,kv,vv,lens,lens,backend='tensorcore'),{'L':L,'H':h,'Hkv':hk,'D':d,'dtype':'bf16'})
for d in (16,64):
    B,T,H=2,32,4
    a=torch.randn(B,T,H,d,device=dev);k=torch.randn_like(a);v=torch.randn_like(a);g=-torch.rand(B,T,H,device=dev)*.2;beta=torch.rand(B,T,H,device=dev);state=torch.randn(B,H,d,d,device=dev)
    bench('delta_chunkwise',lambda:q.delta_chunkwise(a,k,v,g,beta,state,chunk_size=16),{'B':B,'T':T,'H':H,'D':d,'chunk':16})
for L in (64,1024):
    B,T,H,dn,dr,dc,dv=1,4,4,16,8,32,16
    x=torch.randn(B,T,H,dn,device=dev);xr=torch.randn(B,T,H,dr,device=dev);c=torch.randn(B,L,dc,device=dev);r=torch.randn(B,L,dr,device=dev);wk=torch.randn(H,dn,dc,device=dev)/math.sqrt(dn);wv=torch.randn(H,dv,dc,device=dev)/math.sqrt(dc)
    lens=torch.tensor([L],device=dev,dtype=torch.int32);positions=torch.arange(L-T,L,device=dev,dtype=torch.int32).reshape(1,T)
    bench('mla_prefill',lambda:q.mla_prefill(x,xr,c,r,wk,wv,lens,positions),{'B':B,'T':T,'H':H,'L':L,'Dn':dn,'Dr':dr,'Dc':dc,'Dv':dv})
    cache=torch.cat((c,r),dim=-1).reshape(L//16,16,dc+dr);table=torch.arange(L//16,device=dev,dtype=torch.int32).reshape(1,-1)
    bench('mla_decode',lambda:q.mla_decode(x[:,-1:].contiguous(),xr[:,-1:].contiguous(),cache,table,lens,positions[:,-1:].contiguous(),wk,wv),{'B':B,'T':1,'H':H,'L':L,'Dc':dc,'Dr':dr})
    iq=torch.randn(B,1,4,16,device=dev);ik=torch.randn(B,L,16,device=dev);weight=torch.randn(B,1,4,device=dev);pos=positions[:,-1:].contiguous();x1=x[:,-1:].contiguous();xr1=xr[:,-1:].contiguous()
    scores=q.lightning_indexer(iq,ik,weight,lens,pos);indices=q.topk_select(scores,16)
    bench('indexer',lambda:q.lightning_indexer(iq,ik,weight,lens,pos),{'B':1,'T':1,'L':L,'H':4,'D':16})
    bench('topk',lambda:q.topk_select(scores,16),{'B':1,'T':1,'L':L,'K':16})
    bench('sparse_mla',lambda:q.sparse_mla(x1,xr1,cache,table,lens,pos,wk,wv,indices),{'B':1,'T':1,'L':L,'K':16})
    def path():
        scores=q.lightning_indexer(iq,ik,weight,lens,pos);ids=q.topk_select(scores,16)
        return q.sparse_mla(x1,xr1,cache,table,lens,pos,wk,wv,ids)
    bench('dsa_complete',path,{'B':1,'T':1,'L':L,'K':16})
B,T,H,L,dn,dr,dc,dv=1,16,4,256,16,8,32,16
x=torch.randn(B,T,H,dn,device=dev);xr=torch.randn(B,T,H,dr,device=dev);c=torch.randn(B,L,dc,device=dev);r=torch.randn(B,L,dr,device=dev);wk=torch.randn(H,dn,dc,device=dev)/math.sqrt(dn);wv=torch.randn(H,dv,dc,device=dev)/math.sqrt(dc)
lens=torch.tensor([L],device=dev,dtype=torch.int32);positions=torch.arange(L-T,L,device=dev,dtype=torch.int32).reshape(1,T)
bench('mla_prefill_tiled',lambda:q.mla_prefill(x,xr,c,r,wk,wv,lens,positions),{'B':B,'T':T,'H':H,'L':L,'Dn':dn,'Dr':dr,'Dc':dc,'Dv':dv})
a=torch.randn(2,256,4,64,device=dev);gate=torch.randn_like(a);weight=torch.randn(64,device=dev)
bench('gated_attention',lambda:q.gated_attention(a,gate),{'B':2,'T':256,'H':4,'D':64})
bench('delta_output',lambda:q.delta_output(a,gate,weight),{'B':2,'T':256,'H':4,'D':64})
x=torch.randn(2,256,128,device=dev);w=torch.randn(128,4,device=dev);state=torch.randn(2,128,4,device=dev)
bench('causal_conv1d',lambda:q.causal_conv1d(x,w,state),{'B':2,'T':256,'C':128,'K':4})
key=torch.randn(4,8,128,device=dev,dtype=torch.bfloat16);value=torch.randn_like(key)
kcache=torch.zeros(4,16,8,128,device=dev,dtype=key.dtype);vcache=torch.zeros_like(kcache)
table=torch.arange(4,device=dev,dtype=torch.int32).reshape(4,1);pos=torch.full((4,),7,device=dev,dtype=torch.int32)
bench('kv_append',lambda:q.kv_append(key,value,kcache,vcache,table,pos),{'B':4,'H':8,'D':128,'dtype':'bf16'})
out={'root':str(pathlib.Path(args.root).resolve()),'time':datetime.datetime.now().astimezone().isoformat(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'inner':args.inner,'repeats':args.repeats,'results':rows}
pathlib.Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
