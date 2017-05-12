##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Custom CMake rules for PyTorch (a hacky way)
FILE(GLOB TORCH_LIB_HINTS 
	"/anaconda/lib/python3.6/site-packages/torch/lib" 
	"/anaconda2/lib/python3.6/site-packages/torch/lib" 
	"${HOME}/anaconda/lib/python2.7/site-packages/torch/lib"
	"${HOME}/anaconda2/lib/python2.7/site-packages/torch/lib"
)
FIND_PATH(TORCH_BUILD_DIR
	NAMES "THNN.h"
	PATHS "${TORCH_LIB_HINTS}"
)

MESSAGE(STATUS "TORCH_BUILD_DIR: " ${TORCH_BUILD_DIR})

# Find the include files
SET(TORCH_TH_INCLUDE_DIR "${TORCH_BUILD_DIR}/include/TH")
SET(TORCH_THC_INCLUDE_DIR "${TORCH_BUILD_DIR}/include/THC")
SET(TORCH_THC_UTILS_INCLUDE_DIR "$ENV{HOME}/pytorch/torch/lib/THC")

SET(Torch_INSTALL_INCLUDE "${TORCH_BUILD_DIR}/include" ${TORCH_TH_INCLUDE_DIR} ${TORCH_THC_INCLUDE_DIR} ${TORCH_THC_UTILS_INCLUDE_DIR})

# Find the libs. We need to find libraries one by one.
FIND_LIBRARY(THC_LIBRARIES NAMES THC THC.1 PATHS ${TORCH_BUILD_DIR} PATH_SUFFIXES lib)
FIND_LIBRARY(TH_LIBRARIES NAMES TH TH.1 PATHS ${TORCH_BUILD_DIR} PATH_SUFFIXES lib)

