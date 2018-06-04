from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='enclib_gpu',
    ext_modules=[
        CUDAExtension('enclib_gpu', [
            'operator.cpp',
            'encoding_kernel.cu',
            'syncbn_kernel.cu',
            'roi_align_kernel.cu',
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
