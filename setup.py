from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qd-attention",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="qd_attention._C",
            sources=[
                "csrc/binding/torch_ops.cpp",
                "csrc/dense/gqa_decode.cu",
                "csrc/dense/gqa_prefill.cu",
                "csrc/cache/kv_append.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
