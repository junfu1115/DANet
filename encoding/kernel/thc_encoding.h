/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2017
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
#include <THC/THC.h>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#define Encoding_(NAME) TH_CONCAT_4(Encoding_, Real, _, NAME)
#define THCTensor        TH_CONCAT_3(TH,CReal,Tensor)
#define THCTensor_(NAME) TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#ifdef __cplusplus
extern "C" {
#endif

// float
#include "generic/encoding_kernel.h"
#include "THC/THCGenerateFloatType.h"

#include "generic/syncbn_kernel.h"
#include "THC/THCGenerateFloatType.h"

#include "generic/pooling_kernel.h"
#include "THC/THCGenerateFloatType.h"

// double
#include "generic/encoding_kernel.h"
#include "THC/THCGenerateDoubleType.h"

#include "generic/syncbn_kernel.h"
#include "THC/THCGenerateDoubleType.h"

#include "generic/pooling_kernel.h"
#include "THC/THCGenerateDoubleType.h"

#ifdef __cplusplus
}
#endif
