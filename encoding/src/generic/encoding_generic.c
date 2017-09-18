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
