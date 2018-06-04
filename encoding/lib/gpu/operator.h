#include <torch/torch.h>
#include <vector>

at::Tensor ROIAlignForwardCUDA(
  const at::Tensor input,
  const at::Tensor rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sample_ratio);

at::Tensor ROIAlignBackwardCUDA(
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
  const at::Tensor beta_);

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor mean_, 
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_, 
  bool train);

std::vector<at::Tensor> Sum_Square_Forward_CUDA(
  const at::Tensor input_);

at::Tensor Sum_Square_Backward_CUDA(
  const at::Tensor input_,
  const at::Tensor gradSum_,
  const at::Tensor gradSquare_);
