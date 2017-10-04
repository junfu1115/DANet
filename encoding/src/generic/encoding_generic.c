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
#define THC_GENERIC_FILE "generic/encoding_generic.c"
#else

int Encoding_(scaledl2_forward)(THCTensor *SL,  
    THCTensor *X, THCTensor *C,  THCTensor *S)
/*
 * ScaledL2 operation
 */
{
		Encoding_(ScaledL2_Forward)(state, SL, X, C, S);
		/* C function return number of the outputs */
		return 0;
}


int Encoding_(scaledl2_backward)(
    THCTensor *GSL, THCTensor *GX, THCTensor *GC,
    THCTensor *X, THCTensor *C, THCTensor *S)
/*
 * ScaledL2 operation
 */
{
		Encoding_(ScaledL2_Backward)(state, GSL, GX, GC, X, C, S);
		/* C function return number of the outputs */
		return 0;
}


int Encoding_(aggregateE_forward)(THCTensor *E, THCTensor *A,
			THCTensor *X, THCTensor *C)
/*
 * Aggregate operation
 */
{
		Encoding_(AggregateE_Forward)(state, E, A, X, C);
		/* C function return number of the outputs */
		return 0;
}


int Encoding_(aggregateE_backward)(THCTensor *GA, THCTensor *GE, 
		THCTensor *A, THCTensor *X, THCTensor *C)
/*
 * Aggregate backward operation to A
 * G (dl/dR), L (dl/dE), A (assignments)
 */
{
		Encoding_(AggregateE_Backward)(state, GA, GE, A, X, C);
		/* C function return number of the outputs */
		return 0;
}


int Encoding_(aggregate_forward)(THCTensor *E, THCTensor *A,
			THCTensor *R)
/*
 * Aggregate operation
 */
{
		Encoding_(Aggregate_Forward)(state, E, A, R);
		/* C function return number of the outputs */
		return 0;
}

int Encoding_(aggregate_backward)(THCTensor *GA, THCTensor *GR, 
		THCTensor *L, THCTensor *A, THCTensor *R)
/*
 * Aggregate backward operation to A
 * G (dl/dR), L (dl/dE), A (assignments)
 */
{
		Encoding_(Aggregate_Backward)(state, GA, GR, L, A, R);
		/* C function return number of the outputs */
		return 0;
}


int Encoding_(residual_forward)(THCTensor *R, THCTensor *X, THCTensor *D)
/*
 * Residual operation
 */
{
		Encoding_(Residual_Forward)(state, R, X, D);
		/* C function return number of the outputs */
		return 0;
}

int Encoding_(residual_backward)(THCTensor *GR, THCTensor *GX, 
    THCTensor *GD)
/*
 * Residual operation
 */
{
		Encoding_(Residual_Backward)(state, GR, GX, GD);
		/* C function return number of the outputs */
		return 0;
}

int Encoding_(squaresqueeze_forward)(THCTensor *L, THCTensor *R)
/*
 * Residual operation
 */
{
    Encoding_(SquareSqueeze_Forward)(state, L, R);
		/* C function return number of the outputs */
		return 0;
}

int Encoding_(squaresqueeze_backward)(THCTensor *GL, THCTensor *GR, 
    THCTensor *R)
/*
 * Residual operation
 */
{
    Encoding_(SquareSqueeze_Backward)(state, GL, GR, R);
		/* C function return number of the outputs */
		return 0;
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int Encoding_(batchnorm_Forward)(THCTensor *output_, THCTensor *input_, 
        THCTensor *mean_, THCTensor *invstd_,
        THCTensor *gamma_, THCTensor *beta_)
/*
 * 
 */
{
    Encoding_(BatchNorm_Forward)(state, output_, input_, 
        mean_, invstd_, gamma_, beta_);
	/* C function return number of the outputs */
	return 0;
}

int Encoding_(batchnorm_Backward)(THCTensor *gradoutput_, 
        THCTensor *input_, THCTensor *gradinput_, 
        THCTensor *gradgamma_, THCTensor *gradbeta_, THCTensor *mean_, 
        THCTensor *invstd_, THCTensor *gamma_, THCTensor *beta_, 
        THCTensor *gradMean_, THCTensor *gradStd_, int train)
/*
 */
{
    Encoding_(BatchNorm_Backward)(state, gradoutput_, input_, gradinput_, 
        gradgamma_, gradbeta_, mean_, invstd_, gamma_, beta_, gradMean_, gradStd_,
        train);
	/* C function return number of the outputs */
	return 0;
}


int Encoding_(sum_square_Forward)(THCTensor *input_, 
        THCTensor *sum_, THCTensor *square_)
/*
 */
{
    Encoding_(Sum_Square_Forward)(state, input_, sum_, square_);
	/* C function return number of the outputs */
	return 0;
}


int Encoding_(sum_square_Backward)(
        THCTensor *gradInput, THCTensor *input_, 
        THCTensor *gradSum_, THCTensor *gradSquare_)
/*
 */
{
    Encoding_(Sum_Square_Backward)(state, gradInput, input_, gradSum_, 
                                   gradSquare_);
	/* C function return number of the outputs */
	return 0;
}
#endif
