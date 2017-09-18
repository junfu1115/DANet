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
int Encoding_Float_aggregate_backward(THCudaTensor *GA, THCudaTensor *GR, 
		THCudaTensor *L, THCudaTensor *A, THCudaTensor *R);

int Encoding_Float_batchnorm_Forward(THCudaTensor *output_, 
        THCudaTensor *input_, THCudaTensor *mean_, 
        THCudaTensor *invstd_, THCudaTensor *gamma_, THCudaTensor *beta_);

int Encoding_Float_batchnorm_Backward(THCudaTensor *gradoutput_, 
        THCudaTensor *input_, THCudaTensor *gradinput_, 
        THCudaTensor *gradgamma_, THCudaTensor *gradbeta_, 
        THCudaTensor *mean_, THCudaTensor *invstd_, 
        THCudaTensor *gamma_,THCudaTensor *beta_, 
        THCudaTensor *gradMean_, THCudaTensor *gradStd_, int train);

int Encoding_Float_sum_square_Forward(THCudaTensor *input_, 
        THCudaTensor *sum_, THCudaTensor *square_);

void Encoding_Float_sum_square_Backward(
        THCudaTensor *gradInput, THCudaTensor *input_, 
        THCudaTensor *gradSum_, THCudaTensor *gradSquare_);

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

int Encoding_Double_aggregate_forward(THCudaDoubleTensor *E, 
        THCudaDoubleTensor *A, THCudaDoubleTensor *R);

int Encoding_Double_aggregate_backward(THCudaDoubleTensor *GA, 
        THCudaDoubleTensor *GR, THCudaDoubleTensor *L, 
        THCudaDoubleTensor *A, THCudaDoubleTensor *R);

int Encoding_Double_batchnorm_Forward(THCudaDoubleTensor *output_, 
        THCudaDoubleTensor *input_, THCudaDoubleTensor *mean_, 
        THCudaDoubleTensor *invstd_, THCudaDoubleTensor *gamma_, THCudaDoubleTensor *beta_);

int Encoding_Double_batchnorm_Backward(THCudaDoubleTensor *gradoutput_, 
        THCudaDoubleTensor *input_, THCudaDoubleTensor *gradinput_, 
        THCudaDoubleTensor *gradgamma_, THCudaDoubleTensor *gradbeta_, 
        THCudaDoubleTensor *mean_, THCudaDoubleTensor *invstd_, 
        THCudaDoubleTensor *gamma_, THCudaDoubleTensor *beta_, 
        THCudaDoubleTensor *gradMean_, THCudaDoubleTensor *gradStd_, int train);

int Encoding_Double_sum_square_Forward(THCudaDoubleTensor *input_, 
        THCudaDoubleTensor *sum_, THCudaDoubleTensor *square_);

void Encoding_Double_sum_square_Backward(
        THCudaDoubleTensor *gradInput, THCudaDoubleTensor *input_, 
        THCudaDoubleTensor *gradSum_, THCudaDoubleTensor *gradSquare_);
