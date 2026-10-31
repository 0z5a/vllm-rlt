"""Optional native kernels; ordinary installs remain Python-only."""

import os

from setuptools import setup

ext_modules = []
cmdclass = {}
if os.environ.get("VLLM_RLT_BUILD_CUDA") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules.append(
        CUDAExtension(
            "vllm_rlt._C",
            sources=[
                "csrc/torch_bindings.cpp",
                "csrc/activation/silu_and_mul.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(ext_modules=ext_modules, cmdclass=cmdclass)
