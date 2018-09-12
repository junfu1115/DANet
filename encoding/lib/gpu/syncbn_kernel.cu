#include <ATen/ATen.h>
#include <vector>

#include "common.h"
#include "device_tensor.h"

namespace {

template <typename DType, typename Acctype, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(Acctype m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<DType, Acctype> operator()(int batch, int plane, int n) {
    DType g = gradOutput[batch][plane][n];
    DType c = ScalarConvert<Acctype, DType>::to(input[batch][plane][n] - mean);
    return Float2<DType, Acctype>(g, g * c);
  }
  const Acctype mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

template <typename DType, typename Acctype>
struct SumOp {
  __device__ SumOp(DeviceTensor<DType, 3> i) : input(i){}
  __device__ __forceinline__ Float2<DType, Acctype> operator()(int batch, int plane, int n) {
    DType g = input[batch][plane][n];
    return Float2<DType, Acctype>(g, g * g);
  }
  DType mean;
  DeviceTensor<DType, 3> input;
};

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor3>
__device__ T reduce(Op op, DeviceTensor3 tensor, int plane) {
  T sum = (T)0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <typename DType>
__global__ void BatchNorm_Forward_kernel (
  DeviceTensor<DType, 3> output,
  DeviceTensor<DType, 3> input,
  DeviceTensor<DType, 1> mean,
  DeviceTensor<DType, 1> std,
  DeviceTensor<DType, 1> gamma,
  DeviceTensor<DType, 1> beta) {
  int c = blockIdx.x;
  /* main operation */ 
  for (int b = 0; b < input.getSize(0); ++b) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      DType inp = input[b][c][x];
      output[b][c][x] = gamma[c] * (inp - mean[c]) /
        std[c] + beta[c];
    }
  }
}

template <typename DType>
__global__ void BatchNorm_Backward_kernel (
    DeviceTensor<DType, 3> gradoutput,
    DeviceTensor<DType, 3> input,
    DeviceTensor<DType, 3> gradinput,
    DeviceTensor<DType, 1> gradgamma,
    DeviceTensor<DType, 1> gradbeta,
    DeviceTensor<DType, 1> mean,
    DeviceTensor<DType, 1> std,
    DeviceTensor<DType, 1> gamma,
    DeviceTensor<DType, 1> beta,
    DeviceTensor<DType, 1> gradMean, 
    DeviceTensor<DType, 1> gradStd,
    bool train) {
  /* declarations of the variables */
  /* Get the index and channels */ 
  int c = blockIdx.x; 
  /* main operation */ 
  GradOp<DType, DType, DeviceTensor<DType, 3>> g(mean[c], input, gradoutput);
  Float2<DType, DType> res = reduce<Float2<DType, DType>,
    GradOp<DType, DType, DeviceTensor<DType, 3>>,
    DeviceTensor<DType, 3>>(g, gradoutput, c);
  DType gradOutputSum = res.v1;
  DType dotP = res.v2;
  DType invstd = DType(1.0) / std[c];
  DType gradScale = invstd * gamma[c];
  if (train && threadIdx.x == 0) {
    gradMean[c] = - gradOutputSum * gamma[c] * invstd;
    gradStd[c]  = - dotP * gamma[c] * invstd * invstd;
  }
  if (gradinput.numElements() > 0) {
    for (int batch = 0; batch < gradoutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < gradoutput.getSize(2); x += blockDim.x) {
        gradinput[batch][c][x] = gradoutput[batch][c][x] * gradScale;
      }
    }
  }
  if (gradgamma.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradgamma[c] += dotP * invstd;
    }
  }
  if (gradbeta.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradbeta[c] += gradOutputSum;
    }
  }
}


template <typename DType>
__global__ void Sum_Square_Forward_kernel (
    DeviceTensor<DType, 3> input,
    DeviceTensor<DType, 1> sum,
    DeviceTensor<DType, 1> square) {
  int c = blockIdx.x;
  /* main operation */ 
  SumOp<DType, DType> g(input);
  Float2<DType, DType> res = reduce<Float2<DType, DType>,
    SumOp<DType, DType>, DeviceTensor<DType, 3>>(g, input, c);
  DType xsum = res.v1;
  DType xsquare = res.v2;
  if (threadIdx.x == 0) {
    sum[c] = xsum;
    square[c] = xsquare;
  }
}

template <typename DType>
__global__ void Sum_Square_Backward_kernel (
  DeviceTensor<DType, 3> gradInput,
  DeviceTensor<DType, 3> input,
  DeviceTensor<DType, 1> gradSum,
  DeviceTensor<DType, 1> gradSquare) {
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

} // namespcae

at::Tensor BatchNorm_Forward_CUDA(
    const at::Tensor input_, 
    const at::Tensor mean_,
    const at::Tensor std_,
    const at::Tensor gamma_,
    const at::Tensor beta_) {
  auto output_ = at::zeros_like(input_);
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> output = devicetensor<scalar_t, 3>(output_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> mean = devicetensor<scalar_t, 1>(mean_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    /* kernel function */
    BatchNorm_Forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        output, input, mean, std, gamma, beta);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return output_;
}

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
    const at::Tensor gradoutput_,
    const at::Tensor input_,
    const at::Tensor mean_, 
    const at::Tensor std_,
    const at::Tensor gamma_,
    const at::Tensor beta_, 
    bool train) {
  /* outputs*/
  at::Tensor gradinput_ = at::zeros_like(input_);
  at::Tensor gradgamma_ = at::zeros_like(gamma_);
  at::Tensor gradbeta_ = at::zeros_like(beta_);
  at::Tensor gradMean_ = at::zeros_like(mean_);
  at::Tensor gradStd_ = at::zeros_like(std_);
  /* cuda utils*/
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradoutput = devicetensor<scalar_t, 3>(gradoutput_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 3> gradinput = devicetensor<scalar_t, 3>(gradinput_);
    DeviceTensor<scalar_t, 1> gradgamma = devicetensor<scalar_t, 1>(gradgamma_);
    DeviceTensor<scalar_t, 1> gradbeta = devicetensor<scalar_t, 1>(gradbeta_);
    DeviceTensor<scalar_t, 1> mean = devicetensor<scalar_t, 1>(mean_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    DeviceTensor<scalar_t, 1> gradMean = devicetensor<scalar_t, 1>(gradMean_);
    DeviceTensor<scalar_t, 1> gradStd = devicetensor<scalar_t, 1>(gradStd_);
    /* kernel function */
    BatchNorm_Backward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(
      gradoutput, input, gradinput, gradgamma, gradbeta, mean, std, 
      gamma, beta, gradMean, gradStd, train);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradinput_, gradMean_, gradStd_, gradgamma_, gradbeta_};
}

std::vector<at::Tensor> Sum_Square_Forward_CUDA(
    const at::Tensor input_) {
  /* outputs */
  at::Tensor sum_ = input_.type().tensor({input_.size(1)}).zero_();
  at::Tensor square_ = input_.type().tensor({input_.size(1)}).zero_();
  /* cuda utils*/
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> sum = devicetensor<scalar_t, 1>(sum_);
    DeviceTensor<scalar_t, 1> square = devicetensor<scalar_t, 1>(square_);
    /* kernel function */
    Sum_Square_Forward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(input, sum, square);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {sum_, square_};
}

at::Tensor Sum_Square_Backward_CUDA(
    const at::Tensor input_,
    const at::Tensor gradSum_,
    const at::Tensor gradSquare_) {
  /* outputs */
  at::Tensor gradInput_ = at::zeros_like(input_);
  /* cuda utils*/
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradInput = devicetensor<scalar_t, 3>(gradInput_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> gradSum = devicetensor<scalar_t, 1>(gradSum_);
    DeviceTensor<scalar_t, 1> gradSquare =devicetensor<scalar_t, 1>(gradSquare_);
    /* kernel function */
    Sum_Square_Backward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(gradInput, input, gradSum, gradSquare);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return gradInput_;
}
