##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# No longer using manual way to find the library.
if(FALSE)
FILE(GLOB TORCH_LIB_HINTS 
	"/anaconda/lib/python3.6/site-packages/torch/lib" 
	"/anaconda2/lib/python3.6/site-packages/torch/lib" 
	"$ENV{HOME}/anaconda/lib/python2.7/site-packages/torch/lib"
	"$ENV{HOME}/anaconda2/lib/python2.7/site-packages/torch/lib"
)
FIND_PATH(TORCH_BUILD_DIR
	NAMES "THNN.h"
	PATHS "${TORCH_LIB_HINTS}"
)
FIND_LIBRARY(THC_LIBRARIES NAMES THC THC.1 PATHS ${TORCH_BUILD_DIR} PATH_SUFFIXES lib)
FIND_LIBRARY(TH_LIBRARIES NAMES TH TH.1 PATHS ${TORCH_BUILD_DIR} PATH_SUFFIXES lib)
endif()

# Set the envrionment variable via python
SET(TORCH_BUILD_DIR "$ENV{TORCH_BUILD_DIR}")
MESSAGE(STATUS "TORCH_BUILD_DIR: " ${TORCH_BUILD_DIR})

# Find the include files
SET(TORCH_TH_INCLUDE_DIR "${TORCH_BUILD_DIR}/include/TH")
SET(TORCH_THC_INCLUDE_DIR "${TORCH_BUILD_DIR}/include/THC")
SET(TORCH_THC_UTILS_INCLUDE_DIR "$ENV{HOME}/pytorch/torch/lib/THC")

SET(Torch_INSTALL_INCLUDE "${TORCH_BUILD_DIR}/include" ${TORCH_TH_INCLUDE_DIR} ${TORCH_THC_INCLUDE_DIR} ${TORCH_THC_UTILS_INCLUDE_DIR})

# Find the libs. We need to find libraries one by one.
SET(TH_LIBRARIES "$ENV{TH_LIBRARIES}")
SET(THC_LIBRARIES "$ENV{THC_LIBRARIES}")
