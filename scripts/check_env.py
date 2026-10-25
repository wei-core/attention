"""Inspect CUDA build prerequisites; does not alter the local environment."""

import os
import shutil
import subprocess
import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    nvcc = shutil.which("nvcc")
    print(f"nvcc: {nvcc or 'NOT FOUND'}")
    if nvcc:
        completed = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        print("nvcc release:", completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else "ERROR")
    print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST", "(auto-detect)"))
    cuda_available = torch.cuda.is_available()
    print(f"CUDA GPU available: {cuda_available}")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  capability={torch.cuda.get_device_capability(i)}")
    ready = nvcc is not None and torch.version.cuda is not None and cuda_available
    print("Build prerequisite status:", "PRESENT (compatibility still unverified)" if ready else "INCOMPLETE")
    if not ready:
        print("Install a supported CUDA-enabled PyTorch build and CUDA Toolkit >=12.8, then retry.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
