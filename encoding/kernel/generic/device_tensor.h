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
#define THC_GENERIC_FILE "generic/device_tensor.h"
#else
template <int Dim>
THCDeviceTensor<real, Dim> devicetensor(THCState *state, THCTensor *t) {
    if (!t) {
        return THCDeviceTensor<real, Dim>();
    }
    int inDim = THCTensor_(nDimension)(state, t);
    if (inDim == Dim) {
        return toDeviceTensor<real, Dim>(state, t);
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
    return THCDeviceTensor<real, Dim>(THCTensor_(data)(state, t), size);
}

struct Encoding_(Float2)
/*
 * For reduce sum calcualtion of two elements
 */
{
    real v1, v2;
    __device__ Encoding_(Float2)() {}
    __device__ Encoding_(Float2)(real x1, real x2) : v1(x1), v2(x2) {}
    __device__ Encoding_(Float2)(real v) : v1(v), v2(v) {}
    __device__ Encoding_(Float2)(int v) :  v1(v), v2(v) {}
    __device__ Encoding_(Float2)& operator+=(const Encoding_(Float2)& a) 
    {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
    }
};

static __device__ __forceinline__ real Encoding_(rwarpSum)(real val) {
#if CUDA_VERSION >= 9000
    unsigned int mask = 0xffffffff;
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        val += __shfl_xor_sync(mask, val, 1 << i, WARP_SIZE);
    }
#else
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
#endif
    return val;
}

static __device__ __forceinline__ Encoding_(Float2) Encoding_(warpSum)(
    Encoding_(Float2) value) 
{
    value.v1 = Encoding_(rwarpSum)(value.v1);
    value.v2 = Encoding_(rwarpSum)(value.v2);
    return value;
}


#endif
