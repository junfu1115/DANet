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

int Encoding_(aggregate_forward)(THCTensor *E, THCTensor *A,
			THCTensor *X, THCTensor *C)
/*
 * Aggregate operation
 */
{
		Encoding_(Aggregate_Forward)(state, E, A, X, C);
		/* C function return number of the outputs */
		return 0;
}

int Encoding_(aggregate_backward)(THCTensor *GA, THCTensor *GE, 
		THCTensor *A, THCTensor *X, THCTensor *C)
/*
 * Aggregate backward operation to A
 * G (dl/dR), L (dl/dE), A (assignments)
 */
{
		Encoding_(Aggregate_Backward)(state, GA, GE, A, X, C);
		/* C function return number of the outputs */
		return 0;
}

#endif
