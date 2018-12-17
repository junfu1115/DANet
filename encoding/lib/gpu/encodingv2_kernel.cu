#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.h"
#include "device_tensor.h"

namespace {

template<typename DType, typename Acctype>
struct KD2Op {
  __device__ KD2Op(DeviceTensor<DType, 3> x,
                   DeviceTensor<DType, 2> c,
                   DeviceTensor<DType, 2> std) : X(x), C(c), STD(std) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) 
  {
      DType r = (X[b][i][d] - C[k][d]) / STD[k][d];
      return ScalarConvert<DType, Acctype>::to(r * r);
  }
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
};

template<typename DType, typename Acctype>
__global__ void Encoding_Dist_Forward_kernel (
    DeviceTensor<DType, 3> KD,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD) {
  /* declarations of the variables */
  int b, k, i, D;
  /* Get the index and channels */ 
  b = blockIdx.z;
  k = blockIdx.x;
  i = blockIdx.y;
  D = X.getSize(2);
  /* main operation */
  KD2Op<DType, Acctype> g(X, C, STD);
  KD[b][i][k] = reduceD<Acctype>(g, b, i, k, D);;
}

template<typename DType, typename Acctype>
struct EncGradXOp {
  __device__ EncGradXOp(
    DeviceTensor<DType, 3> gkd,
    DeviceTensor<DType, 3> x,
    DeviceTensor<DType, 2> c,
    DeviceTensor<DType, 2> std) : GKD(gkd), X(x), C(c), STD(std) {}
    // DeviceTensor<DType, 1> s, S(s)
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(
      2 * GKD[b][i][k] * (X[b][i][d] - C[k][d]) / 
      (STD[k][d] * STD[k][d]));
  }
  DeviceTensor<DType, 3> GKD;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
  // DeviceTensor<DType, 1> S;
};

template<typename DType, typename Acctype>
__global__ void Encoding_GradX_kernel (
    DeviceTensor<DType, 3> GKD,
    DeviceTensor<DType, 3> GX,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD) {
    // DeviceTensor<DType, 1> S
  /* declarations of the variables */
  int b, d, i, K;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y;
  d = blockIdx.x;
  K = C.getSize(0);
  /* main operation */
  EncGradXOp<DType, Acctype> g(GKD, X, C, STD);
  GX[b][i][d] = reduceK<Acctype>(g, b, i, d, K);
}

template<typename DType, typename Acctype>
struct EncGradSTDOp {
  __device__ EncGradSTDOp(
    DeviceTensor<DType, 3> gkd,
    DeviceTensor<DType, 3> x,
    DeviceTensor<DType, 2> c,
    DeviceTensor<DType, 2> std) : GKD(gkd), X(x), C(c), STD(std) {}
    // DeviceTensor<DType, 1> s, S(s)
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(
      -2 * GKD[b][i][k] * (X[b][i][d] - C[k][d]) *
      (X[b][i][d] - C[k][d]) / (STD[k][d] * STD[k][d] * STD[k][d]));
  }
  DeviceTensor<DType, 3> GKD;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
  // DeviceTensor<DType, 1> S;
};

template<typename DType, typename Acctype>
__global__ void Encoding_GradCSTD_kernel (
    DeviceTensor<DType, 3> GKD,
    DeviceTensor<DType, 2> GC,
    DeviceTensor<DType, 2> GSTD,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD) {
  /* declarations of the variables */
  int k, d, B, N;
  /* Get the index and channels */ 
  d = blockIdx.x;
  k = blockIdx.y;
  B = X.getSize(0);
  N = X.getSize(1);
  /* main operation */
  EncGradXOp<DType, Acctype> g1(GKD, X, C, STD);
  EncGradSTDOp<DType, Acctype> g2(GKD, X, C, STD);
  GC[k][d] = -reduceBN<Acctype>(g1, k, d, B, N);
  GSTD[k][d] += reduceBN<Acctype>(g2, k, d, B, N);
}

template<typename DType, typename Acctype>
struct EncGradSTDXOp {
  __device__ EncGradSTDXOp(
    DeviceTensor<DType, 2> gstd,
    DeviceTensor<DType, 3> x,
    DeviceTensor<DType, 2> c,
    DeviceTensor<DType, 2> std) : GSTD(gstd), X(x), C(c), STD(std) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(
      GSTD[k][d] * (X[b][i][d] - C[k][d]) / STD[k][d]);
  }
  DeviceTensor<DType, 2> GSTD;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
};

template<typename DType, typename Acctype>
__global__ void Encoding_GradSTDX_kernel (
    DeviceTensor<DType, 2> GSTD,
    DeviceTensor<DType, 3> GX,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD,
    int N) {
  /* declarations of the variables */
  int b, d, i, K;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y;
  d = blockIdx.x;
  K = C.getSize(0);
  /* main operation */
  EncGradSTDXOp<DType, Acctype> g(GSTD, X, C, STD);
  GX[b][i][d] += reduceK<Acctype>(g, b, i, d, K) / N;
}

template<typename DType, typename Acctype>
struct AggOpV2 {
  __device__ AggOpV2(DeviceTensor<DType, 3> a,
                     DeviceTensor<DType, 3> x,
                     DeviceTensor<DType, 2> c,
                     DeviceTensor<DType, 2> std) : A(a), X(x), C(c), STD(std) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(A[b][i][k] * (X[b][i][d] - C[k][d]) /
                                             STD[k][d]);
  }
  DeviceTensor<DType, 3> A;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
};

template<typename DType, typename Acctype>
__global__ void AggregateV2_Forward_kernel (
    DeviceTensor<DType, 3> E,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD) {
  /* declarations of the variables */
  int b, k, d, N;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x;
  k = blockIdx.y;
  N = X.getSize(1);
  /* main operation */
  AggOpV2<DType, Acctype> g(A, X, C, STD);
  E[b][k][d] = reduceN<Acctype>(g, b, k, d, N);
}

template<typename DType, typename Acctype>
struct AggV2BackOp {
  __device__ AggV2BackOp(DeviceTensor<DType, 3> g,
                         DeviceTensor<DType, 3> x,
                         DeviceTensor<DType, 2> c,
                         DeviceTensor<DType, 2> std) : G(g), X(x), C(c), STD(std) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(G[b][k][d] * (X[b][i][d] - C[k][d]) /
                                             STD[k][d]);
  }
  DeviceTensor<DType, 3> G;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
  DeviceTensor<DType, 2> STD;
};

template<typename DType, typename Acctype>
__global__ void AggregateV2_Backward_kernel (
    DeviceTensor<DType, 3> GA,
    DeviceTensor<DType, 3> GE,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C,
    DeviceTensor<DType, 2> STD) {
  /* declarations of the variables */
  int b, k, i, D;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y;
  k = blockIdx.x;
  D = GE.getSize(2);
  /* main operation */
  AggV2BackOp<DType, Acctype> g(GE, X, C, STD);
  GA[b][i][k] = reduceD<Acctype>(g, b, i, k, D);
}

} // namespace

at::Tensor Encoding_Dist_Inference_Forward_CUDA(
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor STD_) {
    // const at::Tensor S_,
  // X \in R^{B, N, D}, C \in R^{K, D}, S \in R^K
  auto KD_ = torch::zeros({X_.size(0), X_.size(1), C_.size(0)}, X_.options());
  // E(x), E(x^2)
  int N = X_.size(0) * X_.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(C_.size(0), X_.size(1), X_.size(0));
  dim3 threads(getNumThreads(C_.size(1)));
  // calculate the kernel distance
  AT_DISPATCH_FLOATING_TYPES(X_.type(), "Encoding_Dist_Inference_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> KD = devicetensor<scalar_t, 3>(KD_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    /* kernel function */
    Encoding_Dist_Forward_kernel<scalar_t, scalar_t>
        <<<blocks, threads, 0, stream>>> (KD, X, C, STD);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return KD_;
}

std::vector<at::Tensor> Encoding_Dist_Inference_Backward_CUDA(
    const at::Tensor GKD_,
    const at::Tensor KD_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor STD_) {
  auto GX_ = at::zeros_like(X_);
  auto GC_ = at::zeros_like(C_);
  auto GSTD_ = at::zeros_like(STD_);
  /* kernel function */
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks1(X_.size(2), X_.size(1), X_.size(0));
  dim3 threads1(getNumThreads(C_.size(0)));
  dim3 blocks2(C_.size(1), C_.size(0));
  dim3 threads2(getNumThreads(X_.size(1)));
  int N = X_.size(0) * X_.size(1);
  AT_DISPATCH_FLOATING_TYPES(X_.type(), "Encoding_Dist_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> GKD = devicetensor<scalar_t, 3>(GKD_);
    DeviceTensor<scalar_t, 2> GSTD = devicetensor<scalar_t, 2>(GSTD_);
    DeviceTensor<scalar_t, 3> GX = devicetensor<scalar_t, 3>(GX_);
    DeviceTensor<scalar_t, 2> GC = devicetensor<scalar_t, 2>(GC_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    Encoding_GradX_kernel<scalar_t, scalar_t>
      <<<blocks1, threads1, 0, stream>>> (GKD, GX, X, C, STD);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
    Encoding_GradCSTD_kernel<scalar_t, scalar_t>
      <<<blocks2, threads2, 0, stream>>> (GKD, GC, GSTD, X, C, STD);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
  }));
  return {GX_, GC_, GSTD_};
}

std::vector<at::Tensor> Encoding_Dist_Forward_CUDA(
    const at::Tensor X_,
    const at::Tensor C_,
    double eps) {
    // const at::Tensor S_,
  // X \in R^{B, N, D}, C \in R^{K, D}, S \in R^K
  auto KD_ = torch::zeros({X_.size(0), X_.size(1), C_.size(0)}, X_.options());
  // E(x), E(x^2)
  int N = X_.size(0) * X_.size(1);
  auto SVar_ = (X_.pow(2).sum(0).sum(0).view({1, X_.size(2)}) -
                2 * C_ * X_.sum(0).sum(0).view({1, X_.size(2)})).expand_as(C_) +
               C_.pow(2) * N;
  auto STD_ = at::sqrt(SVar_ / N + eps);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(C_.size(0), X_.size(1), X_.size(0));
  dim3 threads(getNumThreads(C_.size(1)));
  // calculate the kernel distance
  AT_DISPATCH_FLOATING_TYPES(X_.type(), "Encoding_Dist_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> KD = devicetensor<scalar_t, 3>(KD_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    /* kernel function */
    Encoding_Dist_Forward_kernel<scalar_t, scalar_t>
        <<<blocks, threads, 0, stream>>> (KD, X, C, STD);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {KD_, STD_, SVar_ / (N - 1)};
}

std::vector<at::Tensor> Encoding_Dist_Backward_CUDA(
    const at::Tensor GKD_,
    const at::Tensor GSTD_,
    const at::Tensor KD_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor STD_) {
  auto GX_ = at::zeros_like(X_);
  auto GC_ = at::zeros_like(C_);
  auto GSTD2_ = GSTD_.clone();
  /* kernel function */
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks1(X_.size(2), X_.size(1), X_.size(0));
  dim3 threads1(getNumThreads(C_.size(0)));
  dim3 blocks2(C_.size(1), C_.size(0));
  dim3 threads2(getNumThreads(X_.size(1)));
  int N = X_.size(0) * X_.size(1);
  AT_DISPATCH_FLOATING_TYPES(X_.type(), "Encoding_Dist_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> GKD = devicetensor<scalar_t, 3>(GKD_);
    DeviceTensor<scalar_t, 2> GSTD = devicetensor<scalar_t, 2>(GSTD2_);
    DeviceTensor<scalar_t, 3> GX = devicetensor<scalar_t, 3>(GX_);
    DeviceTensor<scalar_t, 2> GC = devicetensor<scalar_t, 2>(GC_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    Encoding_GradX_kernel<scalar_t, scalar_t>
      <<<blocks1, threads1, 0, stream>>> (GKD, GX, X, C, STD);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
    Encoding_GradCSTD_kernel<scalar_t, scalar_t>
      <<<blocks2, threads2, 0, stream>>> (GKD, GC, GSTD, X, C, STD);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
    Encoding_GradSTDX_kernel<scalar_t, scalar_t>
      <<<blocks1, threads1, 0, stream>>> (GSTD, GX, X, C, STD, N);
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
  }));
  // d_sigma/d_c
  GC_ = GC_ - GSTD2_ * (X_.mean(0).mean(0) - C_) / STD_;
  return {GX_, GC_};
}

at::Tensor AggregateV2_Forward_CUDA(
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor STD_) {
  /* Device tensors */
  auto E_ = torch::zeros({A_.size(0), C_.size(0), C_.size(1)}, A_.options());
  // auto IS_ = 1.0f / (S_ + eps).sqrt();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // B, K, D
  dim3 blocks(C_.size(1), C_.size(0), X_.size(0));
  dim3 threads(getNumThreads(X_.size(1)));

  AT_DISPATCH_FLOATING_TYPES(A_.type(), "Aggregate_Forward_CUDA", ([&] {
    DeviceTensor<scalar_t, 3> E = devicetensor<scalar_t, 3>(E_);
    DeviceTensor<scalar_t, 3> A = devicetensor<scalar_t, 3>(A_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    /* kernel function */
    AggregateV2_Forward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>>(E, A, X, C, STD);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return E_;
}

std::vector<at::Tensor> AggregateV2_Backward_CUDA(
    const at::Tensor GE_,
    const at::Tensor E_,
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor STD_) {
  auto gradA_ = at::zeros_like(A_);
  auto gradX_ = at::bmm(A_ , (GE_ / STD_.unsqueeze(0)));
  auto gradC_ = -(A_.sum(1).unsqueeze(2) * GE_ / STD_.unsqueeze(0)).sum(0);
  auto gradSTD_ = -(GE_ * E_).sum(0) / STD_;
  // auto gradS_ = -0.5 * (GE_ * E_).sum(2).sum(0) / S_;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
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
    DeviceTensor<scalar_t, 2> STD = devicetensor<scalar_t, 2>(STD_);
    AggregateV2_Backward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>> (GA, GE, A, X, C, STD);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradA_, gradX_, gradC_, gradSTD_};
}
