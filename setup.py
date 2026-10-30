from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qd-attention",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="attention._C",
            sources=[
                "csrc/binding/torch_ops.cpp",
                "csrc/cuda/dense/gqa_decode.cu",
                "csrc/cuda/dense/gqa_prefill.cu",
                "csrc/cuda/cache/kv_append.cu",
                "csrc/cuda/mla/mla.cu",
                "csrc/cuda/mla/mla_prefill.cu",
                "csrc/cuda/qwen/gates.cu",
                "csrc/cuda/qwen/causal_conv.cu",
                "csrc/cuda/qwen/delta_chunkwise.cu",
                "csrc/cuda/sparse/indexer.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
