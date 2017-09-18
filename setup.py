##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys

from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

extra_compile_args = ['-std=c++11', '-Wno-write-strings']
if os.getenv('PYTORCH_BINARY_BUILD') and platform.system() == 'Linux':
    print('PYTORCH_BINARY_BUILD found. Static linking libstdc++ on Linux')
    extra_compile_args += ['-static-libstdc++']
    extra_link_args += ['-static-libstdc++']

setup(
    name="encoding",
    version="0.0.1",
    description="PyTorch Encoding Layer",
    url="https://github.com/zhanghang1989/PyTorch-Encoding-Layer",
    author="Hang Zhang",
    author_email="zhang.hang@rutgers.edu",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    extra_compile_args=extra_compile_args,
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
