#include <torch/torch.h>
// CPU declarations

at::Tensor ROIAlignForwardCPU(
  const at::Tensor& input,
  const at::Tensor& bottom_rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);

at::Tensor ROIAlignBackwardCPU(
  const at::Tensor& bottom_rois,
  const at::Tensor& grad_output, // gradient of the output of the layer
  int64_t b_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlignForwardCPU, "ROI Align forward (CPU)");
  m.def("roi_align_backward", &ROIAlignBackwardCPU, "ROI Align backward (CPU)");
}
