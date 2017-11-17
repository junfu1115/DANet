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
#define THC_GENERIC_FILE "generic/encoding_utils.c"
#else

struct Encoding_(AggOp) {
    __device__ Encoding_(AggOp)(THCDeviceTensor<real, 3> a,
                                THCDeviceTensor<real, 3> x,
                                THCDeviceTensor<real, 2> c)
        : A(a), X(x), C(c) {}
    __device__ __forceinline__ real operator()(int b, int i, int k, int d) 
    {
        return A[b][i][k].ldg() * (X[b][i][d].ldg()-C[k][d].ldg());
    }
    THCDeviceTensor<real, 3> A;
    THCDeviceTensor<real, 3> X;
    THCDeviceTensor<real, 2> C;
};

__device__ real Encoding_(reduce_agg)(
        Encoding_(AggOp) op, 
        int b, int k, int d, int N)
{
    real sum = 0;
    for (int x = threadIdx.x; x < N; x += blockDim.x) {
        sum += op(b,x,k,d);
    }
    // sum over NumThreads within a warp
    sum = Encoding_(rwarpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ real shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (real) 0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(rwarpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

struct Encoding_(AggBackOp) {
    __device__ Encoding_(AggBackOp)(THCDeviceTensor<real, 3> ge,
                                    THCDeviceTensor<real, 3> x,
                                    THCDeviceTensor<real, 2> c)
        : GE(ge), X(x), C(c) {}
    __device__ __forceinline__ real operator()(int b, int i, int k, int d) 
    {
        return GE[b][k][d].ldg() * (X[b][i][d].ldg()-C[k][d].ldg());
    }
    THCDeviceTensor<real, 3> GE;
    THCDeviceTensor<real, 3> X;
    THCDeviceTensor<real, 2> C;
};

__device__ real Encoding_(reduce_aggback)(
        Encoding_(AggBackOp) op, 
        int b, int i, int k, int D)
{
    real sum = 0;
    for (int x = threadIdx.x; x < D; x += blockDim.x) {
        sum += op(b,i,k,x);
    }
    // sum over NumThreads within a warp
    sum = Encoding_(rwarpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ real shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (real) 0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(rwarpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

struct Encoding_(L2Op) {
    __device__ Encoding_(L2Op)(THCDeviceTensor<real, 3> x,
                               THCDeviceTensor<real, 2> c)
        : X(x), C(c) {}
    __device__ __forceinline__ real operator()(int b, int i, int k, int d) 
    {
        real r = X[b][i][d].ldg() - C[k][d].ldg();
        return r * r;
    }
    THCDeviceTensor<real, 3> X;
    THCDeviceTensor<real, 2> C;
};

__device__ real Encoding_(reduce_sl2)(
        Encoding_(L2Op) op, 
        int b, int i, int k, int D)
{
    real sum = 0;
    for (int x = threadIdx.x; x < D; x += blockDim.x) {
        sum += op(b,i,k,x);
    }
    // sum over NumThreads within a warp
    sum = Encoding_(rwarpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ real shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (real) 0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(rwarpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

struct Encoding_(L2XBackOp) {
    __device__ Encoding_(L2XBackOp)(
        THCDeviceTensor<real, 3> gsl,
        THCDeviceTensor<real, 3> x,
        THCDeviceTensor<real, 2> c,
        THCDeviceTensor<real, 1> s
    ) : GSL(gsl), X(x), C(c), S(s) {}
    __device__ __forceinline__ real operator()(int b, int i, int k, int d) 
    {
        return 2*S[k].ldg() * GSL[b][i][k].ldg() *
            (X[b][i][d].ldg()-C[k][d].ldg());
    }
    THCDeviceTensor<real, 3> GSL;
    THCDeviceTensor<real, 3> X;
    THCDeviceTensor<real, 2> C;
    THCDeviceTensor<real, 1> S;
};

__device__ real Encoding_(reduce_sl2xback)(
        Encoding_(L2XBackOp) op, 
        int b, int i, int d, int K)
{
    real sum = 0;
    for (int x = threadIdx.x; x < K; x += blockDim.x) {
        sum += op(b,i,x,d);
    }
    // sum over NumThreads within a warp
    sum = Encoding_(rwarpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ real shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (real) 0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(rwarpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

struct Encoding_(L2CBackOp) {
    __device__ Encoding_(L2CBackOp)(
        THCDeviceTensor<real, 3> gsl,
        THCDeviceTensor<real, 3> x,
        THCDeviceTensor<real, 2> c,
        THCDeviceTensor<real, 1> s
    ) : GSL(gsl), X(x), C(c), S(s) {}
    __device__ __forceinline__ real operator()(int b, int i, int k, int d) 
    {
        return -2*S[k].ldg() * GSL[b][i][k].ldg() *
                (X[b][i][d].ldg()-C[k][d].ldg());
    }
    THCDeviceTensor<real, 3> GSL;
    THCDeviceTensor<real, 3> X;
    THCDeviceTensor<real, 2> C;
    THCDeviceTensor<real, 1> S;
};

__device__ real Encoding_(reduce_sl2cback)(
        Encoding_(L2CBackOp) op, 
        int k, int d, int B, int N)
{
    real sum = 0;
    for (int batch = 0; batch < B; ++batch) {
        for (int x = threadIdx.x; x < N; x += blockDim.x) {
            sum += op(batch,x,k,d);
        }
    }
    // sum over NumThreads within a warp
    sum = Encoding_(rwarpSum)(sum);

    // 'transpose', and reduce within warp again
    __shared__ real shared[32];

    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        if (threadIdx.x / WARP_SIZE < 32) {
                shared[threadIdx.x / WARP_SIZE] = sum;
        }
    }
    if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
        // zero out the other entries in shared
        shared[threadIdx.x] = (real) 0;
    }
    __syncthreads();
    if (threadIdx.x / WARP_SIZE == 0) {
        sum = Encoding_(rwarpSum)(shared[threadIdx.x]);
        if (threadIdx.x == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

#endif
