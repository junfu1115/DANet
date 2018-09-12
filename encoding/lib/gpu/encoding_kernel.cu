#include <ATen/ATen.h>
#include <vector>

#include "common.h"
#include "device_tensor.h"

namespace {

template<typename DType, typename Acctype>
struct AggOp {
  __device__ AggOp(DeviceTensor<DType, 3> a,
                   DeviceTensor<DType, 3> x,
                   DeviceTensor<DType, 2> c) : A(a), X(x), C(c) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(A[b][i][k] * (X[b][i][d] - C[k][d]));
  }
  DeviceTensor<DType, 3> A;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
};

template<typename DType, typename Acctype>
struct AggBackOp {
  __device__ AggBackOp(DeviceTensor<DType, 3> g,
                       DeviceTensor<DType, 3> x,
                       DeviceTensor<DType, 2> c) : G(g), X(x), C(c) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(G[b][k][d] * (X[b][i][d] - C[k][d]));
  }
  DeviceTensor<DType, 3> G;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
};

template<typename DType, typename Acctype>
struct SL2Op {
  __device__ SL2Op(DeviceTensor<DType, 3> x,
                   DeviceTensor<DType, 2> c) : X(x), C(c) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) 
  {
      DType r = X[b][i][d] - C[k][d];
      return ScalarConvert<DType, Acctype>::to(r * r);
  }
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
};

template<typename DType, typename Acctype>
struct SL2GradXOp {
  __device__ SL2GradXOp(
    DeviceTensor<DType, 3> gsl,
    DeviceTensor<DType, 3> x,
    DeviceTensor<DType, 2> c,
    DeviceTensor<DType, 1> s
  ) : GSL(gsl), X(x), C(c), S(s) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) 
  {
    return ScalarConvert<DType, Acctype>::to(
      2 * S[k] * GSL[b][i][k] * (X[b][i][d]-C[k][d]));
  }
  DeviceTensor<DType, 3> GSL;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 1> S;
};


template<typename T, typename Op>
__device__ T reduceN(
    Op op, int b, int k, int d, int N) {
  T sum = 0;
  for (int x = threadIdx.x; x < N; x += blockDim.x) {
      sum += op(b,x,k,d);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
      if (threadIdx.x / WARP_SIZE < 32) {
              shared[threadIdx.x / WARP_SIZE] = sum;
      }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
      // zero out the other entries in shared
      shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceD(
    Op op, int b, int i, int k, int D) {
  T sum = 0;
  for (int x = threadIdx.x; x < D; x += blockDim.x) {
      sum += op(b,i,k,x);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
      if (threadIdx.x / WARP_SIZE < 32) {
              shared[threadIdx.x / WARP_SIZE] = sum;
      }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
      // zero out the other entries in shared
      shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceK(
    Op op, int b, int i, int d, int K) {
  T sum = 0;
  for (int x = threadIdx.x; x < K; x += blockDim.x) {
    sum += op(b,i,x,d);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    if (threadIdx.x / WARP_SIZE < 32) {
            shared[threadIdx.x / WARP_SIZE] = sum;
    }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceBN(
    Op op, 
    int k, int d, int B, int N) {
  T sum = 0;
  for (int batch = 0; batch < B; ++batch) {
    for (int x = threadIdx.x; x < N; x += blockDim.x) {
        sum += op(batch,x,k,d);
    }
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);
  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    if (threadIdx.x / WARP_SIZE < 32) {
            shared[threadIdx.x / WARP_SIZE] = sum;
    }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
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

template<typename DType, typename Acctype>
__global__ void Aggregate_Forward_kernel (
    DeviceTensor<DType, 3> E,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C) {
  /* declarations of the variables */
  int b, k, d, N;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x;
  k = blockIdx.y;
  N = X.getSize(1);
  /* main operation */
  AggOp<DType, Acctype> g(A,X,C);
  E[b][k][d] = reduceN<Acctype>(g, b, k, d, N);
}

template<typename DType, typename Acctype>
__global__ void Aggregate_Backward_kernel (
    DeviceTensor<DType, 3> GA,
    DeviceTensor<DType, 3> GE,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C) {
  /* declarations of the variables */
  int b, k, i, D;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y;
  k = blockIdx.x;
  D = GE.getSize(2);
  /* main operation */
  AggBackOp<DType, Acctype> g(GE,X,C);
  GA[b][i][k] = reduceD<Acctype>(g, b, i, k, D);
}

template<typename DType, typename Acctype>
__global__ void ScaledL2_Forward_kernel (
    DeviceTensor<DType, 3> SL,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 1> S) {
  /* declarations of the variables */
  int b, k, i, D;
  /* Get the index and channels */ 
  b = blockIdx.z;
  k = blockIdx.x;
  i = blockIdx.y;
  D = X.getSize(2);
  /* main operation */
  SL2Op<DType, Acctype> g(X,C);
  SL[b][i][k] = S[k] * reduceD<Acctype>(g,b,i,k,D);;
}

template<typename DType, typename Acctype>
__global__ void ScaledL2_GradX_kernel (
    DeviceTensor<DType, 3> GSL,
    DeviceTensor<DType, 3> GX,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 1> S) {
  /* declarations of the variables */
  int b, d, i, K;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x;
  i = blockIdx.y;
  K = C.getSize(0);
  /* main operation */
  SL2GradXOp<DType, Acctype> g(GSL,X,C,S);
  GX[b][i][d] = reduceK<Acctype>(g,b,i,d,K);
}

template<typename DType, typename Acctype>
__global__ void ScaledL2_GradC_kernel (
    DeviceTensor<DType, 3> GSL,
    DeviceTensor<DType, 2> GC,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 1> S) {
  /* declarations of the variables */
  int k, d, B, N;
  /* Get the index and channels */ 
  d = blockIdx.x;
  k = blockIdx.y;
  B = X.getSize(0);
  N = X.getSize(1);
  /* main operation */
  SL2GradXOp<DType, Acctype> g(GSL,X,C,S);
  GC[k][d] = - reduceBN<Acctype>(g, k, d, B, N);
}

}// namespace

at::Tensor Aggregate_Forward_CUDA(
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_) {
  /* Device tensors */
  auto E_ = A_.type().tensor({A_.size(0), C_.size(0), C_.size(1)}).zero_(); 
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  // B, K, D
  dim3 blocks(C_.size(1), C_.size(0), X_.size(0));
  dim3 threads(getNumThreads(X_.size(1)));

  AT_DISPATCH_FLOATING_TYPES(A_.type(), "Aggregate_Forward_CUDA", ([&] {
    DeviceTensor<scalar_t, 3> E = devicetensor<scalar_t, 3>(E_);
    DeviceTensor<scalar_t, 3> A = devicetensor<scalar_t, 3>(A_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    /* kernel function */
    Aggregate_Forward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>>(E, A, X, C);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return E_;
}

std::vector<at::Tensor> Aggregate_Backward_CUDA(
    const at::Tensor GE_,
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_) {
  auto gradA_ = at::zeros_like(A_);
  auto gradX_ = at::bmm(A_, GE_);
  auto gradC_ = (-GE_ * A_.sum(1).unsqueeze(2)).sum(0);
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  // B, K, D
  dim3 blocks(C_.size(0), X_.size(1), X_.size(0));
  dim3 threads(getNumThreads(C_.size(1)));
  AT_DISPATCH_FLOATING_TYPES(A_.type(), "Aggregate_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> GA = devicetensor<scalar_t, 3>(gradA_);
    DeviceTensor<scalar_t, 3> GE = devicetensor<scalar_t, 3>(GE_);
    DeviceTensor<scalar_t, 3> A = devicetensor<scalar_t, 3>(A_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    Aggregate_Backward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>> (GA, GE, A, X, C);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradA_, gradX_, gradC_};
}

at::Tensor ScaledL2_Forward_CUDA(
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_) {
  auto SL_ = X_.type().tensor({X_.size(0), X_.size(1), C_.size(0)}).zero_();
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks(C_.size(0), X_.size(1), X_.size(0));
  dim3 threads(getNumThreads(C_.size(1)));

  AT_DISPATCH_FLOATING_TYPES(X_.type(), "ScaledL2_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> SL = devicetensor<scalar_t, 3>(SL_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 1> S = devicetensor<scalar_t, 1>(S_);
    /* kernel function */
    ScaledL2_Forward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>> (SL, X, C, S);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return SL_;
}

std::vector<at::Tensor> ScaledL2_Backward_CUDA(
    const at::Tensor GSL_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_,
    const at::Tensor SL_) {
  auto GX_ = at::zeros_like(X_);
  auto GC_ = at::zeros_like(C_);
  /* kernel function */
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
  dim3 blocks1(X_.size(2), X_.size(1), X_.size(0));
  dim3 threads1(getNumThreads(C_.size(0)));
  dim3 blocks2(C_.size(1), C_.size(0));
  dim3 threads2(getNumThreads(X_.size(1)));
  //std::vector<int> size{ 1, 1, K};
  //auto GS_ = GSL_ * (SL_ / at::_unsafe_view(S_, size))
  auto GS_ = (GSL_ * (SL_ / S_.view({1, 1, C_.size(0)}))).sum(0).sum(0);
  AT_DISPATCH_FLOATING_TYPES(X_.type(), "ScaledL2_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> GSL = devicetensor<scalar_t, 3>(GSL_);
    DeviceTensor<scalar_t, 3> GX = devicetensor<scalar_t, 3>(GX_);
    DeviceTensor<scalar_t, 2> GC = devicetensor<scalar_t, 2>(GC_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 1> S = devicetensor<scalar_t, 1>(S_);
    ScaledL2_GradX_kernel<scalar_t, scalar_t>
      <<<blocks1, threads1, 0, stream>>> (GSL, GX, X, C, S);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
    ScaledL2_GradC_kernel<scalar_t, scalar_t>
      <<<blocks2, threads2, 0, stream>>> (GSL, GC, X, C, S);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
  }));
  return {GX_, GC_, GS_};
}
