from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernels",
            sources=["bindings.cpp", "kernels.cu"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_86",           # Ampere; change for your GPU
                    "--use_fast_math",
                    "-lineinfo",             # useful for Nsight Compute
                    "--generate-code=arch=compute_86,code=sm_86",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
