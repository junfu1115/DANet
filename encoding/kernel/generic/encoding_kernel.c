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


__global__ void Encoding_(AggregateE_Forward_kernel) (
    THCDeviceTensor<real, 3> E,
    THCDeviceTensor<real, 3> A,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> C)
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
    if (d >= E.getSize(2) || k >= E.getSize(1))    return;
    /* main operation */
    for(i=0; i<N; i++) {
        sum += A[b][i][k].ldg() * (X[b][i][d].ldg()-C[k][d].ldg());
    }
    E[b][k][d] = sum;
}

void Encoding_(AggregateE_Forward)(THCState *state, THCTensor *E_, 
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
    dim3 threads(16, 16);
    dim3 blocks(E.getSize(2)/16+1, E.getSize(1)/16+1, 
                            E.getSize(0));
    Encoding_(AggregateE_Forward_kernel)<<<blocks, threads, 0, stream>>>
        (E, A, X, C);
    THCudaCheck(cudaGetLastError());
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
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
    if (d >= E.getSize(2) || k >= E.getSize(1))    return;
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
__global__ void Encoding_(AggregateE_Backward_kernel) (
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
    int b, k, d, i, D;
    real sum;
    /* Get the index and channels */ 
    b = blockIdx.z;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.x * blockDim.x + threadIdx.x;
    D = GE.getSize(2);
    /* boundary check for output G \in R^{BxNxKxD} */
    if (k >= GA.getSize(2) || i >= GA.getSize(1))    return;
    /* main operation */
    sum = 0;
    for(d=0; d<D; d++) {
        sum += GE[b][k][d].ldg() * (X[b][i][d].ldg()-C[k][d].ldg());
    }
    GA[b][i][k] = sum;
}

void Encoding_(AggregateE_Backward)(THCState *state, THCTensor *GA_, 
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
    dim3 threads(16, 16);
    dim3 blocks(GA.getSize(2)/16+1, GA.getSize(1)/16+1, 
                GA.getSize(0));
    Encoding_(AggregateE_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GA, GE, A, X, C);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(Aggregate_Backward_kernel) (
    THCDeviceTensor<real, 3> GA,
    THCDeviceTensor<real, 4> GR,
    THCDeviceTensor<real, 3> GE,
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
    D = GE.getSize(2);
    /* boundary check for output G \in R^{BxNxKxD} */
    if (k >= GR.getSize(2) || i >= GR.getSize(1))    return;
    /* main operation */
    sum = 0;
    for(d=0; d<D; d++) {
        GR[b][i][k][d] = GE[b][k][d].ldg() * A[b][i][k].ldg();
        sum += GE[b][k][d].ldg() * R[b][i][k][d].ldg();
    }
    GA[b][i][k] = sum;
}

void Encoding_(Aggregate_Backward)(THCState *state, THCTensor *GA_, 
     THCTensor *GR_, THCTensor *GE_, THCTensor *A_, THCTensor *R_)
/*
 * aggregate backward to assignment weights
 * G (dl/dR), L (dl/dE), A
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 5, GA_, GR_, GE_, A_, R_);
    if (THCTensor_(nDimension)(state, GA_) != 3 ||
        THCTensor_(nDimension)(state, GR_) != 4 ||
        THCTensor_(nDimension)(state, GE_)  != 3 ||
        THCTensor_(nDimension)(state, A_)  != 3 ||
        THCTensor_(nDimension)(state, R_)  != 4)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> GA = devicetensor<3>(state, GA_);
    THCDeviceTensor<real, 4> GR = devicetensor<4>(state, GR_);
    THCDeviceTensor<real, 3> GE = devicetensor<3>(state, GE_);
    THCDeviceTensor<real, 3> A = devicetensor<3>(state, A_);
    THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(GA.getSize(2)/16+1, GA.getSize(1)/16+1, 
                GA.getSize(0));
    Encoding_(Aggregate_Backward_kernel)<<<blocks, threads, 0, stream>>>(GA,
              GR, GE, A, R);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(Residual_Forward_kernel) (
    THCDeviceTensor<real, 4> R,
    THCDeviceTensor<real, 3> X,
    THCDeviceTensor<real, 2> D)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, d, i, K;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    K = R.getSize(2);
    /* boundary check for output */
    if (d >= X.getSize(2) || i >= X.getSize(1))    return;
    /* main operation */
    for(k=0; k<K; k++) {
        R[b][i][k][d] = X[b][i][d].ldg() - D[k][d].ldg();
    }
}

void Encoding_(Residual_Forward)(
    THCState *state, THCTensor *R_, THCTensor *X_, THCTensor *D_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 3, R_, X_, D_); 
    if (THCTensor_(nDimension)(state, R_) != 4 ||
        THCTensor_(nDimension)(state, X_) != 3 ||
        THCTensor_(nDimension)(state, D_) != 2)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
    THCDeviceTensor<real, 3> X = devicetensor<3>(state, X_);
    THCDeviceTensor<real, 2> D = devicetensor<2>(state, D_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(X.getSize(2)/16+1, X.getSize(1)/16+1, 
                X.getSize(0));
    Encoding_(Residual_Forward_kernel)<<<blocks, threads, 0, stream>>>(R, X, D);
    THCudaCheck(cudaGetLastError());
}


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(ResidualX_Backward_kernel) (
    THCDeviceTensor<real, 4> GR,
    THCDeviceTensor<real, 3> GX)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, d, i, K;
    real sum;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    K = GR.getSize(2);
    /* boundary check for output */
    if (d >= GX.getSize(2) || i >= GX.getSize(1)) return;
    /* main operation */
    sum = 0;
    for(k=0; k<K; k++) {
        sum += GR[b][i][k][d].ldg();
    }
    GX[b][i][d] = sum;
}

__global__ void Encoding_(ResidualD_Backward_kernel) (
    THCDeviceTensor<real, 4> GR,
    THCDeviceTensor<real, 2> GD)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, d, i, B, N;
    real sum;
    /* Get the index and channels */ 
    d = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;
    B = GR.getSize(0);
    N = GR.getSize(1);
    /* boundary check for output */
    if (d >= GD.getSize(1) || k >= GD.getSize(0)) return;
    /* main operation */
    sum = 0;
    for(b=0; b<B; b++) {
        for(i=0; i<N; i++) {
            sum -= GR[b][i][k][d].ldg();
        }
    }
    GD[k][d] = sum;
}

void Encoding_(Residual_Backward)(
    THCState *state, THCTensor *GR_, THCTensor *GX_, THCTensor *GD_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 3, GR_, GX_, GD_); 
    if (THCTensor_(nDimension)(state, GR_) != 4 ||
        THCTensor_(nDimension)(state, GX_) != 3 ||
        THCTensor_(nDimension)(state, GD_) != 2)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 4> GR = devicetensor<4>(state, GR_);
    THCDeviceTensor<real, 3> GX = devicetensor<3>(state, GX_);
    THCDeviceTensor<real, 2> GD = devicetensor<2>(state, GD_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(GX.getSize(2)/16+1, GX.getSize(1)/16+1, 
                GX.getSize(0));
    Encoding_(ResidualX_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GR, GX);
    THCudaCheck(cudaGetLastError());
    dim3 blocks2(GD.getSize(1)/16+1, GD.getSize(0)/16+1); 
    Encoding_(ResidualD_Backward_kernel)<<<blocks2, threads, 0, stream>>>
        (GR, GD);
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
    int b, k, d, i, D;
    real r, sum;
    /* Get the index and channels */ 
    b = blockIdx.z;
    k = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    D = X.getSize(2);
    /* boundary check for output */
    if (k >= SL.getSize(2) || i >= SL.getSize(1)) return;
    /* main operation */
    sum = 0;
    for(d=0; d<D; d++) {
        r = X[b][i][d].ldg() - C[k][d].ldg();
        sum += r * r;
    }
    SL[b][i][k] = S[k] * sum;
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
    dim3 threads(16, 16);
    dim3 blocks(SL.getSize(2)/16+1, SL.getSize(1)/16+1, 
                SL.getSize(0));
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
    int b, k, d, i, K;
    real sum;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    K = C.getSize(0);
    /* boundary check for output */
    if (d >= GX.getSize(2) || i >= GX.getSize(1)) return;
    /* main operation */
    sum = 0;
    for(k=0; k<K; k++) {
        sum += 2*S[k].ldg() * GSL[b][i][k].ldg() *
            (X[b][i][d].ldg()-C[k][d].ldg());
    }
    GX[b][i][d] = sum;
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
    int b, k, d, i, B, N;
    real sum;
    /* Get the index and channels */ 
    d = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;
    B = X.getSize(0);
    N = X.getSize(1);
    /* boundary check for output */
    if (d >= GC.getSize(1) || k >= GC.getSize(0)) return;
    /* main operation */
    sum = 0;
    for(b=0; b<B; b++) {
        for(i=0; i<N; i++) {
            sum += -2*S[k].ldg() * GSL[b][i][k].ldg() *
                (X[b][i][d].ldg()-C[k][d].ldg());
        }
    }
    GC[k][d] = sum;
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
    dim3 threads(16, 16);
    dim3 blocks(GX.getSize(2)/16+1, GX.getSize(1)/16+1, 
                GX.getSize(0));
    Encoding_(ScaledL2X_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GSL, GX, X, C, S);
    THCudaCheck(cudaGetLastError());
    dim3 blocks2(GC.getSize(1)/16+1, GX.getSize(0)/16+1);
    Encoding_(ScaledL2C_Backward_kernel)<<<blocks2, threads, 0, stream>>>
        (GSL, GC, X, C, S);
    THCudaCheck(cudaGetLastError());
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

__global__ void Encoding_(SquareSqueeze_Forward_kernel) (
    THCDeviceTensor<real, 3> L,
    THCDeviceTensor<real, 4> R)
/*
 * aggregating forward kernel function
 */
{
    /* declarations of the variables */
    int b, k, d, i, D;
    real sum;
    /* Get the index and channels */ 
    b = blockIdx.z;
    k = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    D = R.getSize(3);
    /* boundary check for output */
    if (k >= L.getSize(2) || i >= L.getSize(1)) return;
    /* main operation */
    sum = 0;
    for(d=0; d<D; d++) {
        sum += R[b][i][k][d].ldg()*R[b][i][k][d].ldg();
    }
    L[b][i][k] = sum;
}

void Encoding_(SquareSqueeze_Forward)(
    THCState *state, THCTensor *L_,  THCTensor *R_)
/*
 * aggregating forward the residuals with assignment weights
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 2, L_, R_); 
    if (THCTensor_(nDimension)(state, L_) != 3 ||
        THCTensor_(nDimension)(state, R_) != 4)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> L = devicetensor<3>(state, L_);
    THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(L.getSize(2)/16+1, L.getSize(1)/16+1, 
                L.getSize(0));
    Encoding_(SquareSqueeze_Forward_kernel)<<<blocks, threads, 0, stream>>>
        (L, R);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
__global__ void Encoding_(SquareSqueeze_Backward_kernel) (
    THCDeviceTensor<real, 3> GL,
    THCDeviceTensor<real, 4> GR,
    THCDeviceTensor<real, 4> R)
/*
 */
{
    /* declarations of the variables */
    int b, k, d, i, D;
    real scale;
    /* Get the index and channels */ 
    b = blockIdx.z;
    k = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    D = R.getSize(3);
    /* boundary check for output */
    if (k >= R.getSize(2) || i >= R.getSize(1)) return;
    /* main operation */
    scale = GL[b][i][k] * 2;
    for(d=0; d<D; d++) {
        GR[b][i][k][d] = scale * R[b][i][k][d];
    }
}

void Encoding_(SquareSqueeze_Backward)(
    THCState *state, THCTensor *GL_, THCTensor *GR_, THCTensor *R_)
/*
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 3, GL_, GR_, R_); 
    if (THCTensor_(nDimension)(state, GL_) != 3 ||
        THCTensor_(nDimension)(state, GR_) != 4 ||
        THCTensor_(nDimension)(state, R_) != 4)
    THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> GL = devicetensor<3>(state, GL_);
    THCDeviceTensor<real, 4> GR = devicetensor<4>(state, GR_);
    THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(R.getSize(2)/16+1, R.getSize(1)/16+1, 
                R.getSize(0));
    Encoding_(SquareSqueeze_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (GL, GR, R);
    THCudaCheck(cudaGetLastError());
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

__global__ void Encoding_(BatchNorm_Forward_kernel) (
    THCDeviceTensor<real, 3> output,
    THCDeviceTensor<real, 3> input,
    THCDeviceTensor<real, 1> mean,
    THCDeviceTensor<real, 1> invstd,
    THCDeviceTensor<real, 1> gamma,
    THCDeviceTensor<real, 1> beta)
{
    int c = blockIdx.x;
    /* main operation */ 
    for (int b = 0; b < input.getSize(0); ++b) {
        for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
            real inp = input[b][c][x].ldg();
            output[b][c][x] = gamma[c].ldg() * (inp - mean[c].ldg()) * 
                invstd[c].ldg() + beta[c].ldg();
        }
    }
}

void Encoding_(BatchNorm_Forward)(THCState *state, 
        THCTensor *output_, THCTensor *input_, 
        THCTensor *mean_, THCTensor *invstd_,
        THCTensor *gamma_, THCTensor *beta_)
/*
 * batch norm forward function
 * assuming the input is already flaghten
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 6, output_, input_, mean_, invstd_, 
                         gamma_, beta_);
    if (THCTensor_(nDimension)(state, output_) != 3 ||
        THCTensor_(nDimension)(state, input_)  != 3 ||
        THCTensor_(nDimension)(state, mean_)   != 1 ||
        THCTensor_(nDimension)(state, invstd_) != 1 ||
        THCTensor_(nDimension)(state, gamma_)  != 1 ||
        THCTensor_(nDimension)(state, beta_)   != 1)
        THError("BatchNorm2d forward: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
    THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
    THCDeviceTensor<real, 1> mean   = devicetensor<1>(state, mean_);
    THCDeviceTensor<real, 1> invstd    = devicetensor<1>(state, invstd_);
    THCDeviceTensor<real, 1> gamma  = devicetensor<1>(state, gamma_);
    THCDeviceTensor<real, 1> beta   = devicetensor<1>(state, beta_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    Encoding_(BatchNorm_Forward_kernel)<<<blocks, threads, 0, stream>>>(
        output, input, mean, invstd, gamma, beta);
    THCudaCheck(cudaGetLastError());
}

struct Encoding_(Float2){
    real v1, v2;
    __device__ Encoding_(Float2)() {}
    __device__ Encoding_(Float2)(real x1, real x2) : v1(x1), v2(x2) {}
    __device__ Encoding_(Float2)(real v) : v1(v), v2(v) {}
    __device__ Encoding_(Float2)(int v) :  v1(v), v2(v) {}
    __device__ Encoding_(Float2)& operator+=(const Encoding_(Float2)& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

static __device__ __forceinline__ real Encoding_(rwarpSum)(real val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ real values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

static __device__ __forceinline__ Encoding_(Float2) Encoding_(warpSum)(Encoding_(Float2) value) {
  value.v1 = Encoding_(rwarpSum)(value.v1);
  value.v2 = Encoding_(rwarpSum)(value.v2);
  return value;
}

struct Encoding_(GradOp) {
    __device__ Encoding_(GradOp)(real m, THCDeviceTensor<real, 3> i, THCDeviceTensor<real, 3> g)
        : mean(m), input(i), gradOutput(g) {}
    __device__ __forceinline__ Encoding_(Float2) operator()(int batch, int plane, int n) {
        real g = gradOutput[batch][plane][n].ldg();
        real c = input[batch][plane][n].ldg() - mean;
        return Encoding_(Float2)(g, g * c);
    }
    real mean;
    THCDeviceTensor<real, 3> input;
    THCDeviceTensor<real, 3> gradOutput;
};

// Sum across (batch, x/y/z) applying Op() pointwise
__device__ Encoding_(Float2) Encoding_(reduce)(Encoding_(GradOp) op, THCDeviceTensor<real, 3> tensor, int plane) {
    Encoding_(Float2) sum = (Encoding_(Float2))0;
    for (int batch = 0; batch < tensor.getSize(0); ++batch) {
        for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
            sum += op(batch, plane, x);
        }
    }

    // sum over NumThreads within a warp
    sum = Encoding_(warpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ Encoding_(Float2) shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (Encoding_(Float2))0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(warpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

__global__ void Encoding_(BatchNorm_Backward_kernel) (
    THCDeviceTensor<real, 3> gradoutput,
    THCDeviceTensor<real, 3> input,
    THCDeviceTensor<real, 3> gradinput,
    THCDeviceTensor<real, 1> gradgamma,
    THCDeviceTensor<real, 1> gradbeta,
    THCDeviceTensor<real, 1> mean,
    THCDeviceTensor<real, 1> invstd,
    THCDeviceTensor<real, 1> gamma,
    THCDeviceTensor<real, 1> beta,
    THCDeviceTensor<real, 1> gradMean, 
    THCDeviceTensor<real, 1> gradStd,
    int train)
{
    /* declarations of the variables */
    /* Get the index and channels */ 
    int c = blockIdx.x; 
    /* main operation */ 
    Encoding_(GradOp) g(mean[c], input, gradoutput);
    Encoding_(Float2) res = Encoding_(reduce)(g, gradoutput, c);
    real gradOutputSum = res.v1;
    real dotP = res.v2;

    //real projScale = dotP * norm * invstd[c].ldg() * invstd[c].ldg();
    real gradScale = invstd[c].ldg() * gamma[c].ldg();
    if (train && threadIdx.x == 0) {
        gradMean[c] = - gradOutputSum * gamma[c].ldg() * invstd[c].ldg();
        gradStd[c]  = - dotP * gamma[c].ldg() * invstd[c].ldg() * invstd[c].ldg();
    }

    if (gradinput.numElements() > 0) {
        for (int batch = 0; batch < gradoutput.getSize(0); ++batch) {
            for (int x = threadIdx.x; x < gradoutput.getSize(2); x += blockDim.x) {
                gradinput[batch][c][x] = gradoutput[batch][c][x].ldg() * gradScale;
            }
        }
    }

    if (gradgamma.numElements() > 0) {
        if (threadIdx.x == 0) {
            gradgamma[c] += dotP * invstd[c].ldg();
        }
    }

    if (gradbeta.numElements() > 0) {
        if (threadIdx.x == 0) {
            gradbeta[c] += gradOutputSum;
        }
    }
}

void Encoding_(BatchNorm_Backward)(THCState *state, 
        THCTensor *gradoutput_, THCTensor *input_, THCTensor *gradinput_, 
        THCTensor *gradgamma_, THCTensor *gradbeta_, THCTensor *mean_, 
        THCTensor *invstd_, THCTensor *gamma_, THCTensor *beta_, 
        THCTensor *gradMean_, THCTensor *gradStd_, int train)
/*
 * batch norm backward function
 * assuming the input is already flaghten
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 6, gradoutput_, input_, gradinput_, 
        gradgamma_, gradbeta_, mean_, invstd_, gamma_, beta_);
    if (THCTensor_(nDimension)(state, gradoutput_) != 3 ||
        THCTensor_(nDimension)(state, input_)      != 3 ||
        THCTensor_(nDimension)(state, gradinput_)  != 3 ||
        THCTensor_(nDimension)(state, gradgamma_)  != 1 ||
        THCTensor_(nDimension)(state, gradbeta_)   != 1 ||
        THCTensor_(nDimension)(state, mean_)   != 1 ||
        THCTensor_(nDimension)(state, invstd_) != 1 ||
        THCTensor_(nDimension)(state, gamma_)  != 1 ||
        THCTensor_(nDimension)(state, beta_)   != 1 || 
        THCTensor_(nDimension)(state, gradMean_) != 1 ||
        THCTensor_(nDimension)(state, gradStd_)  != 1 )
        THError("BatchNorm2d backward: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> gradoutput = 
        devicetensor<3>(state, gradoutput_);
    THCDeviceTensor<real, 3> input = 
        devicetensor<3>(state, input_);
    THCDeviceTensor<real, 3> gradinput = 
        devicetensor<3>(state, gradinput_);
    THCDeviceTensor<real, 1> gradgamma = 
        devicetensor<1>(state, gradgamma_);
    THCDeviceTensor<real, 1> gradbeta = devicetensor<1>(state, gradbeta_);
    THCDeviceTensor<real, 1> mean     = devicetensor<1>(state, mean_);
    THCDeviceTensor<real, 1> invstd   = devicetensor<1>(state, invstd_);
    THCDeviceTensor<real, 1> gamma    = devicetensor<1>(state, gamma_);
    THCDeviceTensor<real, 1> beta     = devicetensor<1>(state, beta_);
    THCDeviceTensor<real, 1> gradMean = devicetensor<1>(state, gradMean_);
    THCDeviceTensor<real, 1> gradStd  = devicetensor<1>(state, gradStd_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    Encoding_(BatchNorm_Backward_kernel)<<<blocks, threads, 0, stream>>>(
        gradoutput, input, gradinput, gradgamma, gradbeta, mean, invstd, 
        gamma, beta, gradMean, gradStd, train);
    THCudaCheck(cudaGetLastError());
}

struct Encoding_(SumOp) {
    __device__ Encoding_(SumOp)(THCDeviceTensor<real, 3> i)
        : input(i){}
    __device__ __forceinline__ Encoding_(Float2) operator()(int batch, int plane, int n) {
        real g = input[batch][plane][n].ldg();
        return Encoding_(Float2)(g, g * g);
    }
    real mean;
    THCDeviceTensor<real, 3> input;
};

// Sum across (batch, x/y/z) applying Op() pointwise
__device__ Encoding_(Float2) Encoding_(reduce_sum)(Encoding_(SumOp) op, THCDeviceTensor<real, 3> tensor, int plane) {
  Encoding_(Float2) sum = (Encoding_(Float2))0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = Encoding_(warpSum)(sum);

  // 'transpose', and reduce within warp again
  __shared__ Encoding_(Float2) shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    if (threadIdx.x / WARP_SIZE < 32) {
        shared[threadIdx.x / WARP_SIZE] = sum;
    }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (Encoding_(Float2))0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = Encoding_(warpSum)(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}


__global__ void Encoding_(Sum_Square_Forward_kernel) (
    THCDeviceTensor<real, 3> input,
    THCDeviceTensor<real, 1> sum,
    THCDeviceTensor<real, 1> square)
{
    int c = blockIdx.x;
    /* main operation */ 
    Encoding_(SumOp) g(input);
    Encoding_(Float2) res = Encoding_(reduce_sum)(g, input, c);
    real xsum = res.v1;
    real xsquare = res.v2;
    
    if (threadIdx.x == 0) {
        sum[c]    = xsum;
        square[c] = xsquare;
    }
}


void Encoding_(Sum_Square_Forward)(THCState *state, 
        THCTensor *input_, THCTensor *sum_, THCTensor *square_)
/*
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 3, input_, sum_, square_);
    if (THCTensor_(nDimension)(state, input_)   != 3 ||
        THCTensor_(nDimension)(state, sum_)     != 1 ||
        THCTensor_(nDimension)(state, square_)  != 1)
        THError("Sum_Square forward: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
    THCDeviceTensor<real, 1> sum    = devicetensor<1>(state, sum_);
    THCDeviceTensor<real, 1> square = devicetensor<1>(state, square_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    Encoding_(Sum_Square_Forward_kernel)<<<blocks, threads, 0, stream>>>(
        input, sum, square);
    THCudaCheck(cudaGetLastError());
}


__global__ void Encoding_(Sum_Square_Backward_kernel) (
    THCDeviceTensor<real, 3> gradInput,
    THCDeviceTensor<real, 3> input,
    THCDeviceTensor<real, 1> gradSum,
    THCDeviceTensor<real, 1> gradSquare)
{
    int c = blockIdx.x;
    /* main operation */ 
    for (int batch = 0; batch < gradInput.getSize(0); ++batch) {
        for (int x = threadIdx.x; x < gradInput.getSize(2); x += blockDim.x)
        {
            gradInput[batch][c][x] = gradSum[c] + 2 * gradSquare[c] *
                input[batch][c][x];
        }
    }   
}


void Encoding_(Sum_Square_Backward)(THCState *state, 
        THCTensor *gradInput_, THCTensor *input_, 
        THCTensor *gradSum_, THCTensor *gradSquare_)
/*
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 4, gradInput_, input_, gradSum_, 
                         gradSquare_);
    if (THCTensor_(nDimension)(state, gradInput_)  != 3 ||
        THCTensor_(nDimension)(state, input_)      != 3 ||
        THCTensor_(nDimension)(state, gradSum_)    != 1 ||
        THCTensor_(nDimension)(state, gradSquare_) != 1)
        THError("Sum_Square forward: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 3> gradInput  = devicetensor<3>(state, gradInput_);
    THCDeviceTensor<real, 3> input      = devicetensor<3>(state, input_);
    THCDeviceTensor<real, 1> gradSum    = devicetensor<1>(state, gradSum_);
    THCDeviceTensor<real, 1> gradSquare =devicetensor<1>(state, gradSquare_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    Encoding_(Sum_Square_Backward_kernel)<<<blocks, threads, 0, stream>>>(
        gradInput, input, gradSum, gradSquare);
    THCudaCheck(cudaGetLastError());
}


#endif
