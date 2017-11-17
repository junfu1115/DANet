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
#define THC_GENERIC_FILE "generic/syncbn_kernel.c"
#else


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

// Sum across (batch, b/c/n) applying Op() pointwise
__device__ Encoding_(Float2) Encoding_(reduce)(
        Encoding_(GradOp) op,
        THCDeviceTensor<real, 3> tensor, 
        int plane) 
{
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
