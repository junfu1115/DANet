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
#define THC_GENERIC_FILE "generic/pooling_kernel.h"
#else

void Encoding_(DilatedAvgPool_Forward)(THCState *state, 
    THCTensor *X_, THCTensor *Y_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);

void Encoding_(DilatedAvgPool_Backward)(THCState *state, 
    THCTensor *gradX_, THCTensor *gradY_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);

#endif
