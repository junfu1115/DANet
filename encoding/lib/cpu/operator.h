#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <vector>

at::Tensor ROIAlign_Forward_CPU(
  const at::Tensor& input,
  const at::Tensor& bottom_rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);

at::Tensor ROIAlign_Backward_CPU(
  const at::Tensor& bottom_rois,
  const at::Tensor& grad_output,
  int64_t b_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);

at::Tensor Aggregate_Forward_CPU(
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C);

std::vector<at::Tensor> Aggregate_Backward_CPU(
    const at::Tensor GE,
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C);

at::Tensor ScaledL2_Forward_CPU(
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_);

std::vector<at::Tensor> ScaledL2_Backward_CPU(
    const at::Tensor GSL_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_,
    const at::Tensor SL_);

at::Tensor BatchNorm_Forward_CPU(
  const at::Tensor input_, 
  const at::Tensor mean_,
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_);

std::vector<at::Tensor> BatchNorm_Backward_CPU(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor mean_, 
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_, 
  bool train);

std::vector<at::Tensor> Sum_Square_Forward_CPU(
  const at::Tensor input_);

at::Tensor Sum_Square_Backward_CPU(
  const at::Tensor input_,
  const at::Tensor gradSum_,
  const at::Tensor gradSquare_);

std::vector<at::Tensor> Non_Max_Suppression_CPU(
  const at::Tensor& input,
  const at::Tensor& scores,
  double thresh);

void CONV_RECTIFY_CPU(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool avg_mode);

// Fused color jitter application
// ctm [4,4], img [H, W, C]
py::array_t<float> apply_transform(int H, int W, int C, py::array_t<float> img, py::array_t<float> ctm) {
  auto img_buf = img.request();
  auto ctm_buf = ctm.request();

  // printf("H: %d, W: %d, C: %d\n", H, W, C);
  py::array_t<float> result{(unsigned long)img_buf.size};
  auto res_buf = result.request();

  float *img_ptr = (float *)img_buf.ptr;
  float *ctm_ptr = (float *)ctm_buf.ptr;
  float *res_ptr = (float *)res_buf.ptr;

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float *ptr = &img_ptr[h * W * C + w * C];
      float *out_ptr = &res_ptr[h * W * C + w * C];
      // manually unroll over C
      out_ptr[0] = ctm_ptr[0] * ptr[0] + ctm_ptr[1] * ptr[1] + ctm_ptr[2] * ptr[2] + ctm_ptr[3];
      out_ptr[1] = ctm_ptr[4] * ptr[0] + ctm_ptr[5] * ptr[1] + ctm_ptr[6] * ptr[2] + ctm_ptr[7];
      out_ptr[2] = ctm_ptr[8] * ptr[0] + ctm_ptr[9] * ptr[1] + ctm_ptr[10] * ptr[2] + ctm_ptr[11];
    }
  }

  result.resize({H, W, C});

  return result;
}
