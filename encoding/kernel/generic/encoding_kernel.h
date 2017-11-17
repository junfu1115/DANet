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
#define THC_GENERIC_FILE "generic/encoding_kernel.h"
#else

void Encoding_(Aggregate_Forward)(THCState *state, THCTensor *E_, 
    THCTensor *A_, THCTensor *X_, THCTensor *C_);

void Encoding_(Aggregate_Backward)(THCState *state, THCTensor *GA_, 
     THCTensor *GE_, THCTensor *A_, THCTensor *X_, THCTensor *C_);

void Encoding_(ScaledL2_Forward)( THCState *state, THCTensor *SL_,  
    THCTensor *X_, THCTensor *C_,  THCTensor *S_);

void Encoding_(ScaledL2_Backward)(
    THCState *state, THCTensor *GSL_, THCTensor *GX_, THCTensor *GC_,
    THCTensor *X_, THCTensor *C_, THCTensor *S_);

#endif
