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

int Encoding_Float_scaledl2_forward(THCudaTensor *SL,  
    THCudaTensor *X, THCudaTensor *C, THCudaTensor *S);

int Encoding_Float_scaledl2_backward(
    THCudaTensor *GSL, THCudaTensor *GX, THCudaTensor *GC,
    THCudaTensor *X, THCudaTensor *C, THCudaTensor *S);

int Encoding_Float_aggregateE_forward(THCudaTensor *E, THCudaTensor *A,
			THCudaTensor *X, THCudaTensor *C);

int Encoding_Float_aggregateE_backward(THCudaTensor *GA, THCudaTensor *GE, 
		THCudaTensor *A, THCudaTensor *X, THCudaTensor *C);

int Encoding_Float_aggregate_forward(THCudaTensor *E, THCudaTensor *A,
			THCudaTensor *R);

int Encoding_Float_aggregate_backward(THCudaTensor *GA, THCudaTensor *GR, 
		THCudaTensor *L, THCudaTensor *A, THCudaTensor *R);

int Encoding_Float_residual_forward(THCudaTensor *R, THCudaTensor *X, 
		THCudaTensor *D);

int Encoding_Float_residual_backward(THCudaTensor *GR, THCudaTensor *GX, 
		THCudaTensor *GD);

int Encoding_Float_squaresqueeze_forward(THCudaTensor *L, THCudaTensor *R);

int Encoding_Float_squaresqueeze_backward(THCudaTensor *GL, 
    THCudaTensor *GR, THCudaTensor *R);

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

int Encoding_Float_sum_square_Backward(
    THCudaTensor *gradInput, THCudaTensor *input_, 
    THCudaTensor *gradSum_, THCudaTensor *gradSquare_);

int Encoding_Float_DilatedAvgPool2d_Forward(
    THCudaTensor *X_, THCudaTensor *Y_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);

int Encoding_Float_DilatedAvgPool2d_Backward(
    THCudaTensor *gradX_, THCudaTensor *gradY_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

int Encoding_Double_scaledl2_forward(THCudaDoubleTensor *SL,  
    THCudaDoubleTensor *X, THCudaDoubleTensor *C,  THCudaDoubleTensor *S);

int Encoding_Double_scaledl2_backward(
    THCudaDoubleTensor *GSL, THCudaDoubleTensor *GX, 
    THCudaDoubleTensor *GC, THCudaDoubleTensor *X, 
    THCudaDoubleTensor *C, THCudaDoubleTensor *S);

int Encoding_Double_aggregateE_forward(THCudaDoubleTensor *E, 
    THCudaDoubleTensor *A, THCudaDoubleTensor *X, THCudaDoubleTensor *C);

int Encoding_Double_aggregateE_backward(THCudaDoubleTensor *GA, 
    THCudaDoubleTensor *GE, THCudaDoubleTensor *A, THCudaDoubleTensor *X, 
    THCudaDoubleTensor *C);

int Encoding_Double_aggregate_forward(
    THCudaDoubleTensor *E, THCudaDoubleTensor *A, THCudaDoubleTensor *R);

int Encoding_Double_aggregate_backward(
    THCudaDoubleTensor *GA, THCudaDoubleTensor *GR, THCudaDoubleTensor *L, 
    THCudaDoubleTensor *A, THCudaDoubleTensor *R);

int Encoding_Double_residual_forward(
    THCudaDoubleTensor *R, THCudaDoubleTensor *X, THCudaDoubleTensor *D);

int Encoding_Double_residual_backward(
    THCudaDoubleTensor *GR, THCudaDoubleTensor *GX, 
    THCudaDoubleTensor *GD);

int Encoding_Double_squaresqueeze_forward(THCudaDoubleTensor *L,
    THCudaDoubleTensor *R);

int Encoding_Double_squaresqueeze_backward(THCudaDoubleTensor *GL, 
    THCudaDoubleTensor *GR, THCudaDoubleTensor *R);

int Encoding_Double_batchnorm_Forward(THCudaDoubleTensor *output_, 
    THCudaDoubleTensor *input_, THCudaDoubleTensor *mean_, 
    THCudaDoubleTensor *invstd_, THCudaDoubleTensor *gamma_, 
    THCudaDoubleTensor *beta_);

int Encoding_Double_batchnorm_Backward(THCudaDoubleTensor *gradoutput_, 
    THCudaDoubleTensor *input_, THCudaDoubleTensor *gradinput_, 
    THCudaDoubleTensor *gradgamma_, THCudaDoubleTensor *gradbeta_, 
    THCudaDoubleTensor *mean_, THCudaDoubleTensor *invstd_, 
    THCudaDoubleTensor *gamma_, THCudaDoubleTensor *beta_, 
    THCudaDoubleTensor *gradMean_, THCudaDoubleTensor *gradStd_, 
    int train);

int Encoding_Double_sum_square_Forward(THCudaDoubleTensor *input_, 
    THCudaDoubleTensor *sum_, THCudaDoubleTensor *square_);

void Encoding_Double_sum_square_Backward(
    THCudaDoubleTensor *gradInput, THCudaDoubleTensor *input_, 
    THCudaDoubleTensor *gradSum_, THCudaDoubleTensor *gradSquare_);

int Encoding_Double_DilatedAvgPool2d_Forward(
    THCudaDoubleTensor *X_, THCudaDoubleTensor *Y_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);

int Encoding_Double_DilatedAvgPool2d_Backward(
    THCudaDoubleTensor *gradX_, THCudaDoubleTensor *gradY_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW);
