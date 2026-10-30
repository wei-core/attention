"""Compare public operator paths with equivalent mathematical work on one GPU."""
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
import triton
import attention as qda
from data import make_decode_case


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)


def measure(fn, warmup, repeats, inner):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    vals=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize()
        vals.append(a.elapsed_time(b)*1000/inner)
    return {'median_us':statistics.median(vals),'min_us':min(vals),'max_us':max(vals),'samples_us':vals}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--warmup',type=int,default=10); p.add_argument('--repeats',type=int,default=5); p.add_argument('--inner',type=int,default=10); args=p.parse_args()
    if not torch.cuda.is_available() or not qda.is_cuda_extension_loaded(): raise RuntimeError('GPU and CUDA extension required')
    torch.manual_seed(123)
    rows=[]
    def run(group,name,fn,case):
        r={'group':group,'name':name,**case,**measure(fn,args.warmup,args.repeats,args.inner)}
        rows.append(r); print(json.dumps(r),flush=True)
    for b in (1,4,16):
        for length in (128,1024,4096):
            q,k,v,table,lens=make_decode_case(lengths=[length]*b,q_heads=32,kv_heads=8,head_dim=128,block_size=16,dtype=torch.bfloat16,device='cuda')
            case={'batch':b,'length':length,'dtype':'bf16','hq':32,'hkv':8,'head_dim':128,'page':16}
            work_cuda=torch.empty((b,32,4,130),device='cuda',dtype=torch.float32)
            work_triton=torch.empty_like(work_cuda)
            funcs={
                'cuda_baseline':lambda:qda.gqa_decode(q,k,v,table,lens),
                'cuda_split4_workspace':lambda:qda.gqa_decode_split(q,k,v,table,lens,splits=4,workspace=work_cuda),
                'triton_split4_workspace':lambda:qda.gqa_decode_split(q,k,v,table,lens,splits=4,workspace=work_triton,backend='triton'),
                'auto':lambda:qda.gqa_decode_auto(q,k,v,table,lens),
            }
            def gather():
                return (k[table.long()].reshape(b,-1,8,128)[:,:length].transpose(1,2).contiguous(),
                        v[table.long()].reshape(b,-1,8,128)[:,:length].transpose(1,2).contiguous())
            kk,vv=gather()
            def sdpa(): return F.scaled_dot_product_attention(q.unsqueeze(2),kk,vv,enable_gqa=True).squeeze(2)
            def gather_sdpa():
                x,y=gather()
                return F.scaled_dot_product_attention(q.unsqueeze(2),x,y,enable_gqa=True).squeeze(2)
            ref=sdpa()
            for name,fn in funcs.items():
                torch.testing.assert_close(fn(),ref,atol=.06,rtol=.06)
                run('decode',name,fn,case)
            run('decode','sdpa_contiguous',sdpa,case)
            run('decode','paged_gather',gather,case)
            run('decode','sdpa_with_gather',gather_sdpa,case)
    from flash_attn import flash_attn_func
    for length in (128,512,1024):
        b,h,hk,d=1,8,2,64
        q=torch.randn(b,length,h,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b,length,hk,d,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        lens=torch.full((b,),length,device='cuda',dtype=torch.int32)
        case={'batch':b,'length':length,'dtype':'bf16','hq':h,'hkv':hk,'head_dim':d,'causal':True}
        funcs={
            'cuda_simt':lambda:qda.gqa_prefill(q,k,v,lens,lens),
            'triton_tensorcore':lambda:qda.gqa_prefill(q,k,v,lens,lens,backend='tensorcore'),
            'sdpa':lambda:F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,enable_gqa=True).transpose(1,2),
            'flash_attention':lambda:flash_attn_func(q,k,v,causal=True),
        }
        ref=funcs['sdpa']()
        for name,fn in funcs.items():
            torch.testing.assert_close(fn(),ref,atol=.06,rtol=.06)
            run('prefill',name,fn,case)
    for b in (1,4,16,64):
        length=128
        _,k,v,table,lens=make_decode_case(lengths=[length]*b,q_heads=8,kv_heads=8,head_dim=128,block_size=16,dtype=torch.bfloat16,device='cuda')
        pos=torch.full((b,),length-1,device='cuda',dtype=torch.int32)
        key=torch.randn(b,8,128,device='cuda',dtype=torch.bfloat16); value=torch.randn_like(key)
        idx=torch.arange(b,device='cuda'); page=table[idx,(pos//16).long()].long(); slot=(pos%16).long()
        def cuda(): return qda.kv_append(key,value,k,v,table,pos)
        def torch_indexed():
            k[page,slot]=key
            v[page,slot]=value
        cuda(); torch.testing.assert_close(k[page,slot],key); torch.testing.assert_close(v[page,slot],value)
        torch_indexed(); torch.testing.assert_close(k[page,slot],key); torch.testing.assert_close(v[page,slot],value)
        case={'batch':b,'dtype':'bf16','hkv':8,'head_dim':128,'page':16,'token_per_sequence':1}
        run('kv_append','cuda',cuda,case)
        run('kv_append','torch_indexed_precomputed_indices',torch_indexed,case)
    import flash_attn
    meta={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'device':torch.cuda.get_device_name(),
          'capability':torch.cuda.get_device_capability(), 'driver_and_gpu':command('nvidia-smi','--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,power.draw,memory.used','--format=csv'),
          'torch':torch.__version__,'torch_cuda':torch.version.cuda,'triton':triton.__version__,'flash_attention':flash_attn.__version__,'python':platform.python_version(),
          'git_head':command('git','rev-parse','HEAD'),'git_status':command('git','status','--short'),
          'warmup':args.warmup,'repeats':args.repeats,'inner':args.inner,'seed':123,'timing':'CUDA events; operator calls, outputs allocated except preallocated workspaces; includes host-induced GPU idle gaps'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'environment':meta,'results':rows},indent=2)+'\n')

if __name__=='__main__': main()
