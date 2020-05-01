#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_Forward_CPU, "ROI Align forward (CPU)");
  m.def("roi_align_backward", &ROIAlign_Backward_CPU, "ROI Align backward (CPU)");
  m.def("aggregate_forward", &Aggregate_Forward_CPU, "Aggregate forward (CPU)");
  m.def("aggregate_backward", &Aggregate_Backward_CPU, "Aggregate backward (CPU)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CPU, "ScaledL2 forward (CPU)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CPU, "ScaledL2 backward (CPU)");
  m.def("batchnorm_forward", &BatchNorm_Forward_CPU, "BatchNorm forward (CPU)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CPU, "BatchNorm backward (CPU)");
  m.def("sumsquare_forward", &Sum_Square_Forward_CPU, "SumSqu forward (CPU)");
  m.def("sumsquare_backward", &Sum_Square_Backward_CPU, "SumSqu backward (CPU)");
  m.def("non_max_suppression", &Non_Max_Suppression_CPU, "NMS (CPU)");
  m.def("conv_rectify", &CONV_RECTIFY_CPU, "Convolution Rectifier (CPU)");
  // Apply fused color jitter
  m.def("apply_transform", &apply_transform, "apply_transform");
}
