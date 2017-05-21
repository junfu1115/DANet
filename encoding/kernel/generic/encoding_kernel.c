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
#define THC_GENERIC_FILE "generic/encoding_kernel.c"
#else

__global__ void Encoding_(Aggregate_Forward_kernel) (
	THCDeviceTensor<real, 3> E,
	THCDeviceTensor<real, 3> A,
	THCDeviceTensor<real, 4> R)
/*
 * aggregating forward kernel function
 */
{
  /* declarations of the variables */
  int b, k, d, i, N;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x * blockDim.x + threadIdx.x;
  k = blockIdx.y * blockDim.y + threadIdx.y;
	N = A.getSize(1);
	/* boundary check for output */
	sum = 0;
	if (d >= E.getSize(2) || k >= E.getSize(1))	return;
	/* main operation */
	for(i=0; i<N; i++) {
		sum += A[b][i][k].ldg() * R[b][i][k][d].ldg();
	}
	E[b][k][d] = sum;
}

void Encoding_(Aggregate_Forward)(THCState *state, THCTensor *E_, 
							THCTensor *A_, THCTensor *R_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
	/* Check the GPU index and tensor dims*/
	THCTensor_(checkGPU)(state, 3, E_, A_, R_);
	if (THCTensor_(nDimension)(state, E_) != 3 ||
			THCTensor_(nDimension)(state, A_) != 3 ||
			THCTensor_(nDimension)(state, R_) != 4)
		THError("Encoding: incorrect input dims. \n");
	/* Device tensors */
	THCDeviceTensor<real, 3> E = devicetensor<3>(state, E_);
	THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
	THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(E.getSize(2)/16+1, E.getSize(1)/16+1, 
							E.getSize(0));
	Encoding_(Aggregate_Forward_kernel)<<<blocks, threads, 0, stream>>>(E, A, R);
	THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(Aggregate_Backward_kernel) (
	THCDeviceTensor<real, 3> GA,
	THCDeviceTensor<real, 4> GR,
	THCDeviceTensor<real, 3> L,
	THCDeviceTensor<real, 3> A,
	THCDeviceTensor<real, 4> R)
/*
 * aggregating backward kernel function
 * G (dl/dR), L (dl/dE), A
 */
{
  /* declarations of the variables */
  int b, k, d, i, D;
	real sum;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.x * blockDim.x + threadIdx.x;
	D = L.getSize(2);
	/* boundary check for output G \in R^{BxNxKxD} */
	if (k >= GR.getSize(2) || i >= GR.getSize(1))	return;
	/* main operation */
	sum = 0;
	for(d=0; d<D; d++) {
		//sum += L[b][k][d].ldg() * R[b][i][k][d].ldg();
		GR[b][i][k][d] = L[b][k][d].ldg() * A[b][i][k].ldg();
		sum += L[b][k][d].ldg() * R[b][i][k][d].ldg();
	}
	GA[b][i][k] = sum;
}

void Encoding_(Aggregate_Backward)(THCState *state, THCTensor *GA_, 
 	THCTensor *GR_, THCTensor *L_, THCTensor *A_, THCTensor *R_)
/*
 * aggregate backward to assignment weights
 * G (dl/dR), L (dl/dE), A
 */
{
	/* Check the GPU index and tensor dims*/
	THCTensor_(checkGPU)(state, 5, GA_, GR_, L_, A_, R_);
	if (THCTensor_(nDimension)(state, GA_) != 3 ||
			THCTensor_(nDimension)(state, GR_) != 4 ||
			THCTensor_(nDimension)(state, L_)  != 3 ||
			THCTensor_(nDimension)(state, A_)  != 3 ||
			THCTensor_(nDimension)(state, R_)  != 4)
		THError("Encoding: incorrect input dims. \n");
	/* Device tensors */
	THCDeviceTensor<real, 3> GA = devicetensor<3>(state, GA_);
	THCDeviceTensor<real, 4> GR = devicetensor<4>(state, GR_);
	THCDeviceTensor<real, 3> L = devicetensor<3>(state, L_);
	THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
	THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(GA.getSize(2)/16+1, GA.getSize(1)/16+1, 
							GA.getSize(0));
	Encoding_(Aggregate_Backward_kernel)<<<blocks, threads, 0, stream>>>(GA,
					GR, L, A, R);
	THCudaCheck(cudaGetLastError());
}
#endif
