#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// The number of cuda threads to use. 512 is used for backward compatibility
constexpr int ROI_CUDA_NUM_THREADS = 512;

// The maximum number of blocks to use in the default kernel call.
constexpr int ROI_MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int ROI_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + ROI_CUDA_NUM_THREADS - 1) / ROI_CUDA_NUM_THREADS,
          ROI_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void RoIAlignForwardKernel(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T* w1,
    T* w2,
    T* w3,
    T* w4,
    int* x_low,
    int* x_high,
    int* y_low,
    int* y_high,
    const int /*index*/ /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    *w1 = *w2 = *w3 = *w4 = 0.;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);

  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = (T)*y_low;
  } else {
    *y_high = *y_low + 1;
  }

  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = (T)*x_low;
  } else {
    *x_high = *x_low + 1;
  }

  T ly = y - *y_low;
  T lx = x - *x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <>
inline __device__ double gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return val;
}

template <typename T>
__global__ void RoIAlignBackwardKernel(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            &w1,
            &w2,
            &w3,
            &w4,
            &x_low,
            &x_high,
            &y_low,
            &y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          /*
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
          */
          gpu_atomic_add(
              static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          gpu_atomic_add(
              static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        }  // if
      }  // ix
    }  // iy
  }  // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward
} // namespace


at::Tensor ROIAlign_Forward_CUDA(
    const at::Tensor input,
    const at::Tensor rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio) {

  AT_ASSERT(input.is_contiguous());
  AT_ASSERT(rois.is_contiguous());
  AT_ASSERT(input.ndimension() == 4);
  AT_ASSERT(rois.ndimension() == 2);
  AT_ASSERT(rois.size(1) == 5);

  auto proposals = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  // Output Tensor is (num_rois, C, pooled_height, pooled_width)
  auto output = torch::zeros({proposals, channels, pooled_height, pooled_width}, input.options());

  auto count = output.numel();
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_Forward_CUDA", ([&] {
    RoIAlignForwardKernel<scalar_t>
      <<<ROI_GET_BLOCKS(count),
         ROI_CUDA_NUM_THREADS,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          count,
          input.data<scalar_t>(),
          static_cast<scalar_t>(spatial_scale),
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          sampling_ratio,
          rois.data<scalar_t>(),
          output.data<scalar_t>());
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return output;
}

at::Tensor ROIAlign_Backward_CUDA(
    const at::Tensor rois,
    const at::Tensor grad_output,
    int64_t b_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio) {

  AT_ASSERT(rois.is_contiguous());
  AT_ASSERT(rois.ndimension() == 2);
  AT_ASSERT(rois.size(1) == 5);

  auto roi_cols = rois.size(1);
  AT_ASSERT(roi_cols == 4 || roi_cols == 5);

  // Output Tensor is (num_rois, C, pooled_height, pooled_width)
  // gradient wrt input features
  auto grad_in = torch::zeros({b_size, channels, height, width}, rois.options());
  auto num_rois = rois.size(0);
  auto count = grad_output.numel();

  AT_DISPATCH_FLOATING_TYPES(rois.type(), "ROIAlign_Backward_CUDA", ([&] {
    RoIAlignBackwardKernel<scalar_t>
      <<<ROI_GET_BLOCKS(count),
         ROI_CUDA_NUM_THREADS,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          count,
          grad_output.data<scalar_t>(),
          num_rois,
          static_cast<scalar_t>(spatial_scale),
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          sampling_ratio,
          grad_in.data<scalar_t>(),
          rois.data<scalar_t>());
  }));

  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return grad_in;
}
