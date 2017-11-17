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
#define THC_GENERIC_FILE "generic/encoding_generic.h"
#else

int Encoding_(scaledl2_forward)(THCTensor *SL,  
    THCTensor *X, THCTensor *C,  THCTensor *S);

int Encoding_(scaledl2_backward)(
    THCTensor *GSL, THCTensor *GX, THCTensor *GC,
    THCTensor *X, THCTensor *C, THCTensor *S);

int Encoding_(aggregate_forward)(THCTensor *E, THCTensor *A,
			THCTensor *X, THCTensor *C);

int Encoding_(aggregate_backward)(THCTensor *GA, THCTensor *GE, 
		THCTensor *A, THCTensor *X, THCTensor *C);

int Encoding_(aggregateP_forward)(THCTensor *E, THCTensor *A,
			THCTensor *R);

int Encoding_(aggregateP_backward)(THCTensor *GA, THCTensor *GR, 
		THCTensor *L, THCTensor *A, THCTensor *R);

int Encoding_(residual_forward)(THCTensor *R, THCTensor *X, THCTensor *D);

int Encoding_(residual_backward)(THCTensor *GR, THCTensor *GX, 
    THCTensor *GD);

int Encoding_(squaresqueeze_forward)(THCTensor *L, THCTensor *R);

int Encoding_(squaresqueeze_backward)(THCTensor *GL, THCTensor *GR, 
    THCTensor *R);

#endif
