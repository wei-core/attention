"""Repeatable end-to-end operator timing for e5bd77f versus Tensor Core changes."""
import argparse,datetime,json,math,pathlib,statistics,sys
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--wmma',action='store_true');args=p.parse_args()
sys.path.insert(0,str(pathlib.Path(args.root).resolve()))
import torch
import qd_attention as q
torch.manual_seed(137);device='cuda';rows=[]
def bench(name,fn,shape):
 for _ in range(5):fn()
 torch.cuda.synchronize(); samples=[]
 for _ in range(7):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(10):fn()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)*100)
 rows.append({'name':name,'shape':shape,'median_us':statistics.median(samples),'samples_us':samples});print(name,shape,rows[-1]['median_us'],flush=True)
for C in (32,64):
 for D in (16,64):
  b,t,h=2,C,4;x=torch.randn(b,t,h,D,device=device);k=torch.randn_like(x);v=torch.randn_like(x);g=-torch.rand(b,t,h,device=device)*.1;beta=torch.rand(b,t,h,device=device);s=torch.randn(b,h,D,D,device=device)
  bench('delta_chunkwise',lambda:q.delta_chunkwise(x,k,v,g,beta,s,chunk_size=C,**({"backend":"wmma"} if args.wmma else {})),{'B':b,'T':t,'H':h,'D':D,'chunk':C,'dtype':'fp32'})
for T in (16,64):
 for L in (64,256):
  b,h,dn,dr,dc,dv=1,4,32,16,64,32;dtype=torch.bfloat16
  x=torch.randn(b,T,h,dn,device=device,dtype=dtype);qr=torch.randn(b,T,h,dr,device=device,dtype=dtype);c=torch.randn(b,L,dc,device=device,dtype=dtype);r=torch.randn(b,L,dr,device=device,dtype=dtype);wk=torch.randn(h,dn,dc,device=device,dtype=dtype)/math.sqrt(dn);wv=torch.randn(h,dv,dc,device=device,dtype=dtype)/math.sqrt(dc);lengths=torch.tensor([L],device=device,dtype=torch.int32);pos=torch.arange(L-T,L,device=device,dtype=torch.int32).reshape(b,T)
  bench('mla_prefill',lambda:q.mla_prefill(x,qr,c,r,wk,wv,lengths,pos,**({"backend":"wmma"} if args.wmma else {})),{'B':b,'T':T,'H':h,'L':L,'Dn':dn,'Dr':dr,'Dc':dc,'Dv':dv,'dtype':'bf16'})
for L in (512,1024,4096):
 b,h,dn,dr,dc,dv=1,4,32,16,64,32;dtype=torch.bfloat16
 x=torch.randn(b,1,h,dn,device=device,dtype=dtype);qr=torch.randn(b,1,h,dr,device=device,dtype=dtype);cache=torch.randn(L//16,16,dc+dr,device=device,dtype=dtype);table=torch.arange(L//16,device=device,dtype=torch.int32).reshape(b,-1);wk=torch.randn(h,dn,dc,device=device,dtype=dtype)/math.sqrt(dn);wv=torch.randn(h,dv,dc,device=device,dtype=dtype)/math.sqrt(dc);lengths=torch.tensor([L],device=device,dtype=torch.int32);pos=torch.tensor([[L-1]],device=device,dtype=torch.int32)
 bench('mla_decode',lambda:q.mla_decode(x,qr,cache,table,lengths,pos,wk,wv),{'B':b,'T':1,'H':h,'L':L,'Dn':dn,'Dr':dr,'Dc':dc,'Dv':dv,'dtype':'bf16'})
out={'root':str(pathlib.Path(args.root).resolve()),'time':datetime.datetime.now().astimezone().isoformat(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'samples':7,'inner':10,'warmup':5,'results':rows}
pathlib.Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
