#include <torch/extension.h>
#include <ATen/ATen.h>
//#include <omp.h>

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>* pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc->at(pre_calc_index) = pc;
            pre_calc_index += 1;
            continue;
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
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc->at(pre_calc_index) = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForwardCompute(
    const int nthreads,
    const T* bottom_data,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    int roi_cols,
    T* top_data) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        &pre_calc);

    int c;
#pragma omp parallel for private(c) \
num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                  pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] +
                  pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[index] = output_val;
        }  // for pw
      }  // for ph
    }  // for c
  }  // for n
}


template <typename T>
void bilinear_interpolate_gradient(
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

  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <class T>
inline void add(const T& val, T* address) {
  *address += val;
}

template <typename T>
void ROIAlignBackwardCompute(
    const int nthreads,
    const T* top_diff,
    const int /*num_rois*/,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois,
    int rois_cols) {
  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * rois_cols;
    int roi_batch_ind = 0;
    if (rois_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
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

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
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
          // atomic add is not needed for now since it is single threaded
          add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        }  // if
      }  // ix
    }  // iy
  }  // for
}  // ROIAlignBackward


at::Tensor ROIAlign_Forward_CPU(
  const at::Tensor& input,
  const at::Tensor& bottom_rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio) {

  AT_ASSERT(input.is_contiguous());
  AT_ASSERT(bottom_rois.is_contiguous());
  AT_ASSERT(input.ndimension() == 4);
  AT_ASSERT(bottom_rois.ndimension() == 2);
  AT_ASSERT(bottom_rois.size(1) == 5);

  // ROIs is the set of region proposals to process. It is a 2D at::Tensor where the first
  // dim is the # of proposals, and the second dim is the proposal itself in the form
  // [batch_index startW startH endW endH]

  auto num_rois = bottom_rois.size(0);
  auto roi_cols = bottom_rois.size(1);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  AT_ASSERT(roi_cols == 4 || roi_cols == 5);

  // Output at::Tensor is (num_rois, C, pooled_height, pooled_width)
  auto output = torch::zeros({num_rois, channels, pooled_height, pooled_width}, input.options());

  AT_ASSERT(input.is_contiguous());
  AT_ASSERT(bottom_rois.is_contiguous());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_Forward_CPU", ([&] {
    ROIAlignForwardCompute<scalar_t>(
      output.numel(),
      input.data<scalar_t>(),
      static_cast<scalar_t>(spatial_scale),
      channels,
      height,
      width,
      pooled_height,
      pooled_width, 
      sampling_ratio,
      bottom_rois.data<scalar_t>(),
      roi_cols,
      output.data<scalar_t>());
  }));

  return output;
}


at::Tensor ROIAlign_Backward_CPU(
  const at::Tensor& bottom_rois,
  const at::Tensor& grad_output, // gradient of the output of the layer
  int64_t b_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio)
{
  AT_ASSERT(bottom_rois.is_contiguous());
  AT_ASSERT(bottom_rois.ndimension() == 2);
  AT_ASSERT(bottom_rois.size(1) == 5);

  auto num_rois = bottom_rois.size(0);
  auto roi_cols = bottom_rois.size(1);

  AT_ASSERT(roi_cols == 4 || roi_cols == 5);

  // Output at::Tensor is (num_rois, C, pooled_height, pooled_width)
  auto grad_in = torch::zeros({b_size, channels, height, width}, bottom_rois.options());

  AT_ASSERT(bottom_rois.is_contiguous());

  AT_DISPATCH_FLOATING_TYPES(bottom_rois.type(), "ROIAlign_Backward_CPU", ([&] {
    ROIAlignBackwardCompute<scalar_t>(
      grad_output.numel(), 
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
      bottom_rois.data<scalar_t>(),
      roi_cols);
  }));

  return grad_in;
}
