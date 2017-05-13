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
from torch.utils.ffi import create_extension

package_base = os.path.dirname(torch.__file__)
this_file = os.path.dirname(os.path.realpath(__file__))

include_path = [os.path.join(os.environ['HOME'],'pytorch/torch/lib/THC'), 
								os.path.join(package_base,'lib/include/ENCODING'), 
								os.path.join(this_file,'encoding/src/')]

sources = ['encoding/src/encoding_lib.cpp']
headers = ['encoding/src/encoding_lib.h']
defines = [('WITH_CUDA', None)]
with_cuda = True 

if platform.system() == 'Darwin':
	ENCODING_LIB = os.path.join(package_base, 'lib/libENCODING.dylib')
else:
	ENCODING_LIB = os.path.join(package_base, 'lib/libENCODING.so')

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
			make_relative_rpath(os.path.join(package_base, 'lib')),
			ENCODING_LIB,
		],
)

if __name__ == '__main__':
    ffi.build()
