"""Derive useful-work/ideal-logical-byte models from preserved operator timings.

These are NOT hardware-counter FLOPs or DRAM traffic. See the report for the
work, data-reuse, masking and compute-ceiling assumptions and limitations.
"""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/all_operators_20261029'
OUT=ROOT/'results/attention_roofline_20261029'
BANDWIDTH=448.0 # decimal GB/s, nominal RTX 5060
FP32_PEAK=19200.0 # GFLOP/s: 3840 scalar cores * 2 FMA FLOP * 2.50 GHz

def load(name):return json.loads((DATA/name).read_text())
DENSE=[load('dense_run1.json'),load('dense_run2.json')]
STAGE=[load('stage345_single_run1.json'),load('stage345_single_run2.json')]

def duration(runs, predicate, nested=None):
    times=[]
    for run in runs:
        matches=[x for x in run['results'] if predicate(x)]
        assert len(matches)==1, matches
        row=matches[0]
        times.append((row[nested] if nested else row)['median_us'])
    return statistics.median(times)

rows=[]
def add(name,case,us,tokens,flops,bytes_,work, byte_model, kind='scalar',float_valid=True):
    r={'operator':name,'case':case,'latency_us':us,'work_items_per_second':tokens*1e6/us,
       'nominal_main_flops':flops,'ideal_unique_bytes':bytes_,'work_model':work,'byte_model':byte_model,'compute_path':kind}
    if flops is None or bytes_ is None:
        r.update(useful_gflops=None,model_ai_flop_per_byte=None,model_bandwidth_ceiling_gflops=None,model_fp32_cuda_core_roof_gflops=None,logical_gbytes_per_s=bytes_/us/1000 if bytes_ else None)
    else:
        r['useful_gflops']=flops/us/1000
        r['model_ai_flop_per_byte']=flops/bytes_ if bytes_ else None
        r['model_bandwidth_ceiling_gflops']=BANDWIDTH*flops/bytes_ if bytes_ else None
        # Not a measured roof: valid as a separate comparison only for scalar FP32-style paths.
        r['model_fp32_cuda_core_roof_gflops']=min(FP32_PEAK,r['model_bandwidth_ceiling_gflops']) if kind=='scalar' and bytes_ else None
        r['logical_gbytes_per_s']=bytes_/us/1000 if bytes_ else None
    rows.append(r)

for b in (1,4,16):
 for l in (128,1024,4096):
  h,hk,d,p=32,8,128,16
  f=4*b*h*l*d
  by=2*(2*b*h*d+2*b*l*hk*d)+4*(b*(l//p)+b)
  for name in ('cuda_baseline','cuda_split4_workspace','triton_split4_workspace','auto'):
   us=duration(DENSE,lambda r,n=name,b=b,l=l:r['group']=='decode' and r['name']==n and r['batch']==b and r['length']==l)
   add('gqa_decode/'+name,{'B':b,'Hq':h,'Hkv':hk,'D':d,'L':l,'P':p,'dtype':'bf16'},us,b,f,by,'QK and PV matmul FLOPs only; each MAC=2, excludes softmax','unique Q/K/V/out + table + lengths, ideal reuse; excludes scratch',kind='scalar')
for l in (128,512,1024):
 b,h,hk,d=1,8,2,64; valid=l*(l+1)//2
 f=4*b*h*d*valid
 by=2*(2*b*l*h*d+2*b*l*hk*d)+8*b
 for name in ('cuda_simt','triton_tensorcore'):
  us=duration(DENSE,lambda r,n=name,l=l:r['group']=='prefill' and r['name']==n and r['length']==l)
  add('gqa_prefill/'+name,{'B':b,'Hq':h,'Hkv':hk,'D':d,'T':l,'L':l,'dtype':'bf16','causal':'square'},us,b*l,f,by,'valid causal QK/PV only; no masked tile or softmax work','unique Q/K/V/out + q/kv lengths, ideal reuse',kind='tensor' if name=='triton_tensorcore' else 'scalar')
for b in (1,4,16,64):
 us=duration(DENSE,lambda r,b=b:r['group']=='kv_append' and r['name']=='cuda' and r['batch']==b)
 by=4*b*8*128*2+8*b
 add('kv_append',{'B':b,'Hkv':8,'D':128,'P':16,'dtype':'bf16'},us,b,None,by,'copy operation, no meaningful FLOP count','K/V read+write plus position and page ID',kind='copy')

stagecases={r['operator']:r for r in STAGE[0]['results']}
def st(name):return duration(STAGE,lambda r,n=name:r['operator']==n,'cuda')
def addstage(name,tokens,f,by,work,byte_model,kind='scalar'):
 x=stagecases[name]
 add(name,dict(x['shape'],dtype='fp32'),st(name),tokens,f,by,work,byte_model,kind=kind)
# MLA fixed shape B=1,T=4,H=4,Dn=16,Dr=8,Dc=32,Dv=16,L=64.
B,T,H,Dn,Dr,Dc,Dv,L,P=1,4,4,16,8,32,16,64,16
valid=61+62+63+64
addstage('latent_kv_write',64,None,2*L*(Dc+Dr)*4+L*8,'copy operation, no meaningful FLOP count','latent+rope read and cache write plus int64 slots',kind='copy')
projbytes=(B*T*H*Dn+H*Dn*Dc+B*T*H*Dc)*4
addstage('mla_query_absorb',B*T,2*B*T*H*Dn*Dc,projbytes,'matrix-vector MAC only; each MAC=2','unique input+weight+output once')
vpbytes=(B*T*H*Dc+H*Dv*Dc+B*T*H*Dv)*4
addstage('mla_value_project',B*T,2*B*T*H*Dv*Dc,vpbytes,'matrix-vector MAC only; each MAC=2','unique input+weight+output once')
latbytes=(B*T*H*(2*Dc+Dr)+B*L*(Dc+Dr))*4+4*(B*(L//P)+B+B*T)
addstage('mla_latent_attention',B*T,2*H*valid*(2*Dc+Dr),latbytes,'valid score and latent aggregation MACs; excludes online softmax','unique absorbed Q/rope Q/cache/output + page/lens/positions; excludes repeated loads')
dirbytes=(B*T*H*(Dn+Dr+Dv)+B*L*(Dc+Dr)+H*Dc*(Dn+Dv))*4+4*(B+B*T)
dirf=2*H*valid*(Dc*(Dn+Dv)+Dn+Dr+Dv)
addstage('mla_prefill',B*T,dirf,dirbytes,'source loop reconstructs K/V for each valid pair; MACs only','unique direct input/weights/output + lens/positions; ignores repeated reconstruction traffic')
basebytes=(B*H*(Dn+Dr+Dv)+B*L*(Dc+Dr)+H*Dc*(Dn+Dv))*4+4*(B*(L//P)+B+B)
basef=2*B*H*(Dn*Dc+Dv*Dc)+2*B*H*L*(2*Dc+Dr)
addstage('mla_decode',B,basef,basebytes,'query/value projection + valid absorbed score/aggregation MACs','unique input/cache/weights/output + page/lens/positions; excludes intermediates')
K=16
sparsebytes=(B*H*(Dn+Dr+Dv)+B*K*(Dc+Dr)+H*Dc*(Dn+Dv))*4+4*(B*(L//P)+B+B+B*K)
sparsef=2*B*H*(Dn*Dc+Dv*Dc)+2*B*H*K*(2*Dc+Dr)
addstage('sparse_mla',B,sparsef,sparsebytes,'query/value projections + selected K score/aggregation MACs','selected unique cache positions + input/weight/output/metadata; excludes intermediates')
# Non-MAC special functions are left unquantified, not assigned invented FLOPs.
for name in ('gated_attention','delta_output'):
 elements=2*16*4*32
 by=3*elements*4+(32*4 if name=='delta_output' else 0)
 addstage(name,2*16,None,by,'sigmoid/SiLU, division/rsqrt not assigned standard FLOPs','unique x/gate/output and optional norm weight; excludes intermediates',kind='special')
for name,t in [('causal_conv1d',16),('causal_conv1d_update',1)]:
 b,c,k=2,64,4
 f=2*b*t*c*k
 by=(b*t*c+c*k+2*b*c*k+b*t*c)*4
 addstage(name,b*t,f,by,'depthwise convolution MACs only; excludes SiLU','unique x/weights/state/new state/output once')
b,h,dk,dv=2,4,16,16
recf=7*b*h*dk*dv
recbytes=(2*b*h*dk+2*b*h*dv+2*b*h+2*b*h*dk*dv)*4
addstage('delta_recurrent',b,recf,recbytes,'nominal prediction 2 + state update 3 + readout 2 FLOPs per state element; excludes normalization/gates','unique q/k/v/g/beta/state/new-state/output once')
t=16
chunkbytes=(2*b*t*h*dk+2*b*t*h*dv+2*b*t*h+2*b*h*dk*dv)*4
addstage('delta_chunkwise',b*t,recf*t,chunkbytes,'equivalent recurrent main work, NOT actual chunk algorithm instructions','unique sequence/state/output; excludes chunk scratch and repeated accesses',kind='equivalent')
iB,iT,iH,iD,iL=1,4,4,16,64
indexbytes=(iB*iT*iH*iD+iB*iL*iD+iB*iT*iH+iB*iT*iL)*4+4*(iB+iB*iT)
addstage('lightning_indexer',iB*iT,2*valid*iH*iD,indexbytes,'valid QK dot MACs only; excludes ReLU, head weights and reduction','unique q/k/weights/output + lens/positions once')
addstage('topk_select',1*4,None,(1*4*64+1*4*16)*4,'integer/comparison-based operation; FLOP roofline not meaningful','unique scores and output indices; excludes workspace and repeated scans',kind='integer')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'metrics.json').write_text(json.dumps({'schema_version':1,'bandwidth_gbytes_per_s_nominal':BANDWIDTH,'fp32_scalar_peak_gflops_estimated':FP32_PEAK,'rows':rows},ensure_ascii=False,indent=2)+'\n')
fields=list(rows[0])
with (OUT/'metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
print(f'{len(rows)} modeled cases written to {OUT}')
