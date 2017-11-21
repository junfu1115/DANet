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
import torch
import platform
import subprocess
from torch.utils.ffi import create_extension

lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
cwd = os.path.dirname(os.path.realpath(__file__))
encoding_lib_path = os.path.join(cwd, "encoding", "lib")

# clean the build files
clean_cmd = ['bash', 'clean.sh']
subprocess.check_call(clean_cmd)

# build CUDA library
os.environ['TORCH_BUILD_DIR'] = lib_path
if platform.system() == 'Darwin':
    os.environ['TH_LIBRARIES'] = os.path.join(lib_path,'libATen.1.dylib')
    ENCODING_LIB = os.path.join(cwd, 'encoding/lib/libENCODING.dylib')

else:
    os.environ['TH_LIBRARIES'] = os.path.join(lib_path,'libATen.so.1')
    ENCODING_LIB = os.path.join(cwd, 'encoding/lib/libENCODING.so')

build_all_cmd = ['bash', 'encoding/make.sh']
subprocess.check_call(build_all_cmd, env=dict(os.environ))

# build FFI
sources = ['encoding/src/encoding_lib.cpp']
headers = [
    'encoding/src/encoding_lib.h',
]
defines = [('WITH_CUDA', None)]
with_cuda = True 

include_path = [os.path.join(lib_path, 'include'),
                os.path.join(cwd,'encoding/kernel'), 
                os.path.join(cwd,'encoding/kernel/include'), 
                os.path.join(cwd,'encoding/src/')]

def make_relative_rpath(path):
    if platform.system() == 'Darwin':
        return '-Wl,-rpath,' + path
    else:
        return '-Wl,-rpath,' + path

ffi = create_extension(
    'encoding._ext.encoding_lib',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    include_dirs = include_path,
    extra_link_args = [
        make_relative_rpath(lib_path),
        make_relative_rpath(encoding_lib_path),
        ENCODING_LIB,
    ],
)

if __name__ == '__main__':
    ffi.build()
