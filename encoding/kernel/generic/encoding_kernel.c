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
#define THC_GENERIC_FILE "kernel/generic/encoding_kernel.c"
#else
template <int Dim>
THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }

  int inDim = THCTensor_(nDimension)(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCTensor_(isContiguous)(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<float, Dim>(THCTensor_(data)(state, t), size);
}

__global__ void Encoding_(Aggregate_Forward_kernel) (
	THCDeviceTensor<real, 3> E,
	THCDeviceTensor<real, 3> A,
	THCDeviceTensor<real, 4> R)
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

void Encoding_(Aggregate_Forward)(THCState *state, THCTensor *E_, THCTensor *A_,
							THCTensor *R_)
/*
 * aggregating the residuals with assignment weights
 */
{
	/* Check the GPU index */
	THCTensor_(checkGPU)(state, 3, E_, A_, R_);
	if (THCTensor_(nDimension)(state, E_) != 3 ||
			THCTensor_(nDimension)(state, A_) != 3 ||
			THCTensor_(nDimension)(state, R_) != 4)
		perror("Encoding: incorrect input dims. \n");
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

#endif
