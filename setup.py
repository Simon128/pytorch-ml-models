import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

# python setup.py build_ext --inplace
def get_extensions():
    #"CXX": "g++-12", "CC": "gcc-12"
    #os.environ["CC"] = "gcc-11"
    #os.environ["CXX"] = "g++-11"
    this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pymlmodels")
    extensions_dir = os.path.join(this_dir, "orientedrcnn", "ext")

    main_file = os.path.join(extensions_dir, "vision.cpp")
    source_cpu = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

    sources = [main_file] + source_cpu
    extension = CppExtension

    extra_compile_args = {}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        print("installing cuda version")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "pymlmodels.orientedrcnn._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

setup(
    name="pymlmodels",
    packages=find_packages(),
    install_requires=[
        "opencv-python==4.7.0.72",
        "tensorboard"
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
