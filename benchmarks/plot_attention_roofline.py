"""Render a conceptual unique-byte FP32 SIMD roofline, not profiler data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/attention_roofline_20261029/metrics.json').read_text())
rows=data['rows']; bw=data['bandwidth_gbytes_per_s_nominal']; peak=data['fp32_scalar_peak_gflops_estimated']
selected=[('gqa_decode/cuda_baseline', 'Decode CUDA'),('gqa_decode/triton_split4_workspace','Decode Triton split-4'),('mla_prefill','MLA direct prefill'),('mla_decode','MLA absorbed decode'),('sparse_mla','Sparse MLA'),('delta_recurrent','Delta recurrent'),('delta_chunkwise','Delta chunk (equivalent work)'),('lightning_indexer','Indexer')]
fig,ax=plt.subplots(figsize=(11,7))
x=np.logspace(-2,3,600)
ax.plot(x,np.minimum(bw*x,peak),color='#20242a',linewidth=2,label='Illustrative FP32 SIMD roof: min(448 × AI, 19,200)')
for name,label in selected:
 match=[r for r in rows if r['operator']==name and (not name.startswith('gqa_decode') or (r['case']['B']==1 and r['case']['L']==4096))]
 assert len(match)==1,name
 r=match[0];ax.scatter(r['model_ai_flop_per_byte'],r['useful_gflops'],s=70,label=label)
ax.set(xscale='log',yscale='log',xlabel='Modeled useful FLOPs / ideal unique bytes (FLOP/B)',ylabel='Useful main-work FLOPs / operator elapsed (GFLOP/s)',xlim=(.1,1000),ylim=(.1,30000),title='Illustrative logical-work model — NOT a measured DRAM roofline')
ax.grid(which='both',alpha=.22);ax.legend(loc='lower right',fontsize=8,framealpha=.9)
fig.text(.13,.015,'Operator timing includes allocation / launch gaps; bytes are ideal unique tensors. Tensor Core path excluded.\nCache traffic, actual instruction work and bottlenecks need hardware counters.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,1));out=ROOT/'results/attention_roofline_20261029/roofline_model.png';fig.savefig(out,dpi=180);print(out)
