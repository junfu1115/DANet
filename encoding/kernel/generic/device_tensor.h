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
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/device_tensor.h"
#else
template <int Dim>
THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }
  int inDim = THCTensor_(nDimension)(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }
  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCTensor_(isContiguous)(state, t));
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
  return THCDeviceTensor<float, Dim>(THCTensor_(data)(state, t), size);
}
#endif
