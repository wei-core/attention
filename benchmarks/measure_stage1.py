"""Record repeatable operator CUDA-event measurements and environment metadata."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import qd_attention as qda
from qd_attention.testing import make_decode_case


def output(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--inner', type=int, default=10)
    args = p.parse_args()
    if not torch.cuda.is_available() or not qda.is_cuda_extension_loaded():
        raise RuntimeError('CUDA and extension required')
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    meta = {'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'driver_and_gpu': output('nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,memory.used', '--format=csv'),
            'nvcc': output('nvcc', '--version'), 'git_commit': output('git', 'rev-parse', 'HEAD'),
            'git_status': output('git', 'status', '--short'),
            'build_flags': 'cxx -O3 -std=c++17; nvcc -O3 -std=c++17 --expt-relaxed-constexpr',
            'arch': 'sm_120', 'method': 'CUDA events; operator including output allocation',
            'warmup': args.warmup, 'samples': args.samples, 'inner': args.inner}
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for batch in (1, 4, 16):
            for length in (128, 1024, 4096):
                inputs = make_decode_case(lengths=[length]*batch, q_heads=32, kv_heads=8, head_dim=128, block_size=16, dtype=dtype, device='cuda')
                for _ in range(args.warmup): qda.gqa_decode(*inputs)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.samples):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.inner): qda.gqa_decode(*inputs)
                    end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end)*1000/args.inner)
                row = {'dtype': str(dtype), 'batch': batch, 'length': length, 'hq': 32, 'hkv': 8, 'head_dim': 128, 'block_size':16,
                       'median_us': statistics.median(samples), 'min_us': min(samples), 'max_us': max(samples), 'samples_us': samples}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del inputs
    (directory/'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
    (directory/'raw.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (directory/'benchmark.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

if __name__ == '__main__': main()
