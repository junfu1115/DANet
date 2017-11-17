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
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, d, N;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x;
    k = blockIdx.y;
    N = X.getSize(1);

    /* main operation */
    Encoding_(AggOp) g(A,X,C);
    E[b][k][d] = Encoding_(reduce_agg)(g,b,k,d,N);
}

void Encoding_(Aggregate_Forward)(THCState *state, THCTensor *E_, 
    THCTensor *A_, THCTensor *X_, THCTensor *C_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 4, E_, A_, X_, C_);
    if (THCTensor_(nDimension)(state, E_) != 3 ||
        THCTensor_(nDimension)(state, A_) != 3 ||
        THCTensor_(nDimension)(state, X_) != 3 ||
        THCTensor_(nDimension)(state, C_) != 2)
        THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> E = devicetensor<3>(state, E_);
    THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
    THCDeviceTensor<real, 3> X = devicetensor<3>(state, X_);
    THCDeviceTensor<real, 2> C = devicetensor<2>(state, C_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    // B, K, D
    dim3 blocks(C.getSize(1), C.getSize(0), X.getSize(0));
    // N
    dim3 threads(getNumThreads(X.getSize(1)));
    Encoding_(Aggregate_Forward_kernel)<<<blocks, threads, 0, stream>>>
        (E, A, X, C);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(Aggregate_Backward_kernel) (
    THCDeviceTensor<real, 3> GA,
    THCDeviceTensor<real, 3> GE,
    THCDeviceTensor<real, 3> A,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C)
/*
 * aggregating backward kernel function
 * G (dl/dR), L (dl/dE), A
 */
{
    /* declarations of the variables */
    int b, k, i, D;
    /* Get the index and channels */ 
    b = blockIdx.z;
    i = blockIdx.y;
    k = blockIdx.x;
    D = GE.getSize(2);
    /* main operation */
    Encoding_(AggBackOp) g(GE,X,C);
    GA[b][i][k] = Encoding_(reduce_aggback)(g,b,i,k,D);
}

void Encoding_(Aggregate_Backward)(THCState *state, THCTensor *GA_, 
     THCTensor *GE_, THCTensor *A_, THCTensor *X_, THCTensor *C_)
/*
 * aggregate backward to assignment weights
 * G (dl/dR), L (dl/dE), A
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 5, GA_, GE_, A_, X_, C_);
    if (THCTensor_(nDimension)(state, GA_) != 3 ||
        THCTensor_(nDimension)(state, GE_)  != 3 ||
        THCTensor_(nDimension)(state, A_)  != 3 ||
        THCTensor_(nDimension)(state, X_)  != 3 ||
        THCTensor_(nDimension)(state, C_)  != 2)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> GA = devicetensor<3>(state, GA_);
    THCDeviceTensor<real, 3> GE = devicetensor<3>(state, GE_);
    THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
    THCDeviceTensor<real, 3> X = devicetensor<3>(state, X_);
    THCDeviceTensor<real, 2> C = devicetensor<2>(state, C_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    // B, K, D
    dim3 blocks(C.getSize(0), X.getSize(1), X.getSize(0));
    // N
    dim3 threads(getNumThreads(C.getSize(1)));
    Encoding_(Aggregate_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GA, GE, A, X, C);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(ScaledL2_Forward_kernel) (
    THCDeviceTensor<real, 3> SL,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C,
    THCDeviceTensor<real, 1> S)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, i, D;
    /* Get the index and channels */ 
    b = blockIdx.z;
    k = blockIdx.x;
    i = blockIdx.y;
    D = X.getSize(2);
    /* main operation */
    Encoding_(L2Op) g(X,C);
    SL[b][i][k] = S[k] * Encoding_(reduce_sl2)(g,b,i,k,D);;
}

void Encoding_(ScaledL2_Forward)(
    THCState *state, THCTensor *SL_,  THCTensor *X_,
    THCTensor *C_,  THCTensor *S_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 4, SL_, X_, C_, S_); 
    if (THCTensor_(nDimension)(state, SL_) != 3 ||
        THCTensor_(nDimension)(state, X_) != 3 ||
        THCTensor_(nDimension)(state, C_) != 2 ||
        THCTensor_(nDimension)(state, S_) != 1)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> SL = devicetensor<3>(state, SL_);
    THCDeviceTensor<real, 3> X  = devicetensor<3>(state, X_);
    THCDeviceTensor<real, 2> C  = devicetensor<2>(state, C_);
    THCDeviceTensor<real, 1> S  = devicetensor<1>(state, S_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(C.getSize(0), X.getSize(1), X.getSize(0));
    dim3 threads(getNumThreads(C.getSize(1)));
    Encoding_(ScaledL2_Forward_kernel)<<<blocks, threads, 0, stream>>>
        (SL, X, C, S);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(ScaledL2X_Backward_kernel) (
    THCDeviceTensor<real, 3> GSL,
    THCDeviceTensor<real, 3> GX,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C,
    THCDeviceTensor<real, 1> S)
/*
 */
{
    /* declarations of the variables */
    int b, d, i, K;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x;
    i = blockIdx.y;
    K = C.getSize(0);
    /* main operation */
    Encoding_(L2XBackOp) g(GSL,X,C,S);
    GX[b][i][d] = Encoding_(reduce_sl2xback)(g,b,i,d,K);
}

__global__ void Encoding_(ScaledL2C_Backward_kernel) (
    THCDeviceTensor<real, 3> GSL,
    THCDeviceTensor<real, 2> GC,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C,
    THCDeviceTensor<real, 1> S)
/*
 */
{
    /* declarations of the variables */
    int k, d, B, N;
    /* Get the index and channels */ 
    d = blockIdx.x;
    k = blockIdx.y;
    B = X.getSize(0);
    N = X.getSize(1);
    /* main operation */
    Encoding_(L2CBackOp) g(GSL,X,C,S);
    GC[k][d] = Encoding_(reduce_sl2cback)(g,k,d,B,N);
}

void Encoding_(ScaledL2_Backward)(
    THCState *state, THCTensor *GSL_, THCTensor *GX_, THCTensor *GC_,
    THCTensor *X_, THCTensor *C_, THCTensor *S_)
/*
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 6, GSL_, GX_, GC_, X_, C_, S_); 
    if (THCTensor_(nDimension)(state, GSL_) != 3 ||
        THCTensor_(nDimension)(state, GX_)  != 3 ||
        THCTensor_(nDimension)(state, GC_)  != 2 ||
        THCTensor_(nDimension)(state, X_)   != 3 ||
        THCTensor_(nDimension)(state, C_)   != 2 ||
        THCTensor_(nDimension)(state, S_)   != 1)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> GSL = devicetensor<3>(state, GSL_);
    THCDeviceTensor<real, 3> GX = devicetensor<3>(state, GX_);
    THCDeviceTensor<real, 2> GC = devicetensor<2>(state, GC_);
    THCDeviceTensor<real, 3> X  = devicetensor<3>(state, X_);
    THCDeviceTensor<real, 2> C  = devicetensor<2>(state, C_);
    THCDeviceTensor<real, 1> S = devicetensor<1>(state, S_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(X.getSize(2), X.getSize(1), X.getSize(0));
    dim3 threads(getNumThreads(C.getSize(0)));
    Encoding_(ScaledL2X_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GSL, GX, X, C, S);
    THCudaCheck(cudaGetLastError());
    dim3 blocks2(C.getSize(1), C.getSize(0));
    dim3 threads2(getNumThreads(X.getSize(1)));
    Encoding_(ScaledL2C_Backward_kernel)<<<blocks2, threads2, 0, stream>>>
        (GSL, GC, X, C, S);
    THCudaCheck(cudaGetLastError());
}

#endif
