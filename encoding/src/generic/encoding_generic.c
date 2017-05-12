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

int Encoding_Float_aggregate_forward(THCudaTensor *E, THCudaTensor *A,
			THCudaTensor *R)
/*
 * Aggregate operation
 */
{
	if (THCTensor_(nDimension)(state, E) != 3 ||
			THCTensor_(nDimension)(state, A) != 3 ||
			THCTensor_(nDimension)(state, R) != 4)
		perror("Encoding: incorrect input dims. \n");

	Encoding_(Aggregate_Forward)(state, E, A, R);
	/* C function return number of the outputs */
	return 0;
}

#endif
