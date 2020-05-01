#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> box_encoder(
  const int N_img,
  const at::Tensor& bbox_input,
  const at::Tensor& bbox_offsets,
  const at::Tensor& labels_input,
  const at::Tensor& dbox,
  const float criteria = 0.5);

std::vector<at::Tensor> random_horiz_flip(
  at::Tensor& img,
  at::Tensor& bboxes,
  const at::Tensor& bbox_offsets,
  const float p,
  const bool nhwc);

at::Tensor ROIAlign_Forward_CUDA(
  const at::Tensor input,
  const at::Tensor rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sample_ratio);

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
  int64_t sampling_ratio);

std::vector<at::Tensor> Non_Max_Suppression_CUDA(
  const at::Tensor& input,
  const at::Tensor& scores,
  double thresh);

at::Tensor Aggregate_Forward_CUDA(
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);

std::vector<at::Tensor> Aggregate_Backward_CUDA(
  const at::Tensor GE_,
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);

at::Tensor ScaledL2_Forward_CUDA(
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor S_);

std::vector<at::Tensor> ScaledL2_Backward_CUDA(
  const at::Tensor GSL_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor S_,
  const at::Tensor SL_);

at::Tensor BatchNorm_Forward_CUDA(
  const at::Tensor input_, 
  const at::Tensor mean_,
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

at::Tensor BatchNorm_Forward_Inp_CUDA(
    const at::Tensor input_, 
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> BatchNorm_Inp_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor output_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> Expectation_Forward_CUDA(
  const at::Tensor input_);

at::Tensor Expectation_Backward_CUDA(
  const at::Tensor input_,
  const at::Tensor gradEx_,
  const at::Tensor gradExs_);

at::Tensor Expectation_Inp_Backward_CUDA(
  const at::Tensor gradInput_,
  const at::Tensor output_,
  const at::Tensor gradEx_,
  const at::Tensor gradExs_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

void LeakyRelu_Forward_CUDA(at::Tensor z, float slope);

void LeakyRelu_Backward_CUDA(at::Tensor z, at::Tensor dz, float slope);

void CONV_RECTIFY_CUDA(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool avg_mode);
