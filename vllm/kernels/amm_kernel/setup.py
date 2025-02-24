from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

LIBAMM_PATH = os.environ.get('LIBAMM_PATH', '/path/to/LibAMM')

setup(
    name='amm_linear_cuda',
    ext_modules=[
        CUDAExtension('amm_linear_cuda', [
            'amm_kernel.cu',
            'amm_linear.cpp',
        ],
        include_dirs=[
            os.path.join(LIBAMM_PATH, 'src'),
            os.path.join(LIBAMM_PATH, 'include'),
        ],
        library_dirs=[
            os.path.join(LIBAMM_PATH, 'build/lib'),
        ],
        libraries=['amm'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_70,code=sm_70',
                '-gencode=arch=compute_75,code=sm_75',
                '-gencode=arch=compute_80,code=sm_80',
                '--compiler-options', "'-fPIC'"
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })