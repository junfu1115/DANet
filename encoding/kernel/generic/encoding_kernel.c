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
    if (d >= E.getSize(2) || k >= E.getSize(1))    return;
    sum = 0;
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

#endif
