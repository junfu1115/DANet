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

#include "thc_encoding.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#define Encoding_(NAME) TH_CONCAT_4(Encoding_, Real, _, NAME)
#define THCTensor        TH_CONCAT_3(TH,CReal,Tensor)
#define THCTensor_(NAME) TH_CONCAT_4(TH,CReal,Tensor_,NAME)

template <int Dim>
THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCudaTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }

  int inDim = THCudaTensor_nDimension(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCudaTensor_isContiguous(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<float, Dim>(THCudaTensor_data(state, t), size);
}

#ifdef __cplusplus
extern "C" {
#endif

#include "generic/encoding_kernel.c"
#include "THC/THCGenerateFloatType.h"

#ifdef __cplusplus
}
#endif
