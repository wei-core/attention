"""Compare equivalent native/operator paths; conversion timing is separate."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qd_attention as qda
from qd_attention.testing import make_decode_case


def timing(fn, warmup, samples, inner):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    measurements=[]
    for _ in range(samples):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize()
        measurements.append(a.elapsed_time(b)*1000/inner)
    return {'median_us':statistics.median(measurements),'min_us':min(measurements),'max_us':max(measurements),'samples_us':measurements}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--warmup',type=int,default=10); p.add_argument('--samples',type=int,default=5); p.add_argument('--inner',type=int,default=10); args=p.parse_args()
    torch.manual_seed(42)
    rows=[]
    def measure(name, fn, meta):
        try:
            result=timing(fn,args.warmup,args.samples,args.inner)
            rows.append(dict(name=name,**meta,**result))
            print(json.dumps(rows[-1]),flush=True)
        except (RuntimeError,ImportError) as e:
            rows.append(dict(name=name,**meta,error=str(e)))
            print(json.dumps(rows[-1]),flush=True)
    for batch in (1,4,16):
        for length in (128,1024,4096):
            q,k,v,table,lens=make_decode_case(lengths=[length]*batch,q_heads=32,kv_heads=8,head_dim=128,block_size=16,dtype=torch.bfloat16,device='cuda')
            meta={'batch':batch,'length':length,'dtype':'bf16','hq':32,'hkv':8,'d':128,'page':16}
            def gather():
                kk=k[table.long()].reshape(batch,-1,8,128)[:,:length].transpose(1,2).contiguous()
                vv=v[table.long()].reshape(batch,-1,8,128)[:,:length].transpose(1,2).contiguous()
                return kk,vv
            kk,vv=gather(); qq=q.unsqueeze(2)
            def sdpa(): return F.scaled_dot_product_attention(qq,kk,vv,enable_gqa=True)
            def end_to_end():
                x,y=gather(); return F.scaled_dot_product_attention(qq,x,y,enable_gqa=True)
            ref=qda.gqa_decode(q,k,v,table,lens)
            torch.testing.assert_close(sdpa().squeeze(2),ref,atol=.06,rtol=.06)
            measure('cuda_baseline',lambda: qda.gqa_decode(q,k,v,table,lens),meta)
            for splits in (1,2,4,8,16):
                work=torch.empty((batch,32,splits,130),dtype=torch.float32,device='cuda')
                fn=lambda s=splits,w=work: qda.gqa_decode_split(q,k,v,table,lens,splits=s,workspace=w,backend='triton')
                torch.testing.assert_close(fn(),ref,atol=.06,rtol=.06)
                measure('triton_split_'+str(splits),fn,meta)
            measure('sdpa_native_contiguous',sdpa,meta)
            measure('paged_gather_only',gather,meta)
            measure('sdpa_with_paged_gather',end_to_end,meta)
            del q,k,v,table,lens,kk,vv,ref
    for length in (128,512):
        q=torch.randn(1,length,8,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,length,2,64,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        lens=torch.tensor([length],device='cuda',dtype=torch.int32)
        meta={'batch':1,'length':length,'dtype':'bf16','hq':8,'hkv':2,'d':64,'causal':True}
        sdpa=lambda:F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,enable_gqa=True).transpose(1,2)
        cuda=lambda:qda.gqa_prefill(q,k,v,lens,lens)
        tc=lambda:qda.gqa_prefill(q,k,v,lens,lens,backend='tensorcore')
        expected=sdpa()
        for name,fn in [('prefill_cuda',cuda),('prefill_tensorcore',tc),('prefill_sdpa',sdpa)]:
            torch.testing.assert_close(fn(),expected,atol=.06,rtol=.06)
            measure(name,fn,meta)
        try:
            from flash_attn import flash_attn_func
            flash=lambda:flash_attn_func(q,k,v,causal=True)
            torch.testing.assert_close(flash(),expected,atol=.06,rtol=.06)
            measure('prefill_flash_attention',flash,meta)
        except (ImportError,RuntimeError) as e:
            rows.append(dict(name='prefill_flash_attention',**meta,error=str(e)))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'device':torch.cuda.get_device_name(),'torch':torch.__version__,'warmup':args.warmup,'samples':args.samples,'inner':args.inner,'results':rows},indent=2)+'\n')

if __name__=='__main__': main()
