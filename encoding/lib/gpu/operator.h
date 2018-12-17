#include <torch/extension.h>
#include <vector>

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

at::Tensor Encoding_Dist_Inference_Forward_CUDA(
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor STD_);

std::vector<at::Tensor> Encoding_Dist_Inference_Backward_CUDA(
  const at::Tensor GKD_,
  const at::Tensor KD_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor STD_);

std::vector<at::Tensor> Encoding_Dist_Forward_CUDA(
  const at::Tensor X,
  const at::Tensor C,
  double eps);

std::vector<at::Tensor> Encoding_Dist_Backward_CUDA(
  const at::Tensor GKD_,
  const at::Tensor GSTD_,
  const at::Tensor KD_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor STD_);

at::Tensor AggregateV2_Forward_CUDA(
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor STD_);

std::vector<at::Tensor> AggregateV2_Backward_CUDA(
  const at::Tensor GE_,
  const at::Tensor E_,
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor STD_);

void LeakyRelu_Forward_CUDA(at::Tensor z, float slope);

void LeakyRelu_Backward_CUDA(at::Tensor z, at::Tensor dz, float slope);
