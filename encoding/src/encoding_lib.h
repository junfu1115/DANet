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

//#include <THC/THC.h>

/*
#define Encoding_(NAME) TH_CONCAT_4(Encoding_, Real, _, NAME)
#define THCTensor        TH_CONCAT_3(TH,CReal,Tensor)
#define THCTensor_(NAME) TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#include "generic/encoding_generic.h"
#include "THC/THCGenerateFloatType.h"
*/

int Encoding_Float_aggregate_forward(THCudaTensor *E, THCudaTensor *A,
			THCudaTensor *R);
int Encoding_Float_aggregate_backward(THCudaTensor *G, THCudaTensor *L,
			THCudaTensor *R);
