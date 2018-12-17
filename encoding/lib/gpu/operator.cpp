#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_Forward_CUDA, "ROI Align forward (CUDA)");
  m.def("roi_align_backward", &ROIAlign_Backward_CUDA, "ROI Align backward (CUDA)");
  m.def("non_max_suppression", &Non_Max_Suppression_CUDA, "NMS (CUDA)");
  m.def("aggregate_forward", &Aggregate_Forward_CUDA, "Aggregate forward (CUDA)");
  m.def("aggregate_backward", &Aggregate_Backward_CUDA, "Aggregate backward (CUDA)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CUDA, "ScaledL2 forward (CUDA)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CUDA, "ScaledL2 backward (CUDA)");
  m.def("batchnorm_forward", &BatchNorm_Forward_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_inp_forward", &BatchNorm_Forward_Inp_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("batchnorm_inp_backward", &BatchNorm_Inp_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("expectation_forward", &Expectation_Forward_CUDA, "Expectation forward (CUDA)");
  m.def("expectation_backward", &Expectation_Backward_CUDA, "Expectation backward (CUDA)");
  m.def("expectation_inp_backward", &Expectation_Inp_Backward_CUDA,
        "Inplace Expectation backward (CUDA)");
  m.def("encoding_dist_forward", &Encoding_Dist_Forward_CUDA, "EncDist forward (CUDA)");
  m.def("encoding_dist_backward", &Encoding_Dist_Backward_CUDA, "Assign backward (CUDA)");
  m.def("encoding_dist_inference_forward", &Encoding_Dist_Inference_Forward_CUDA,
        "EncDist Inference forward (CUDA)");
  m.def("encoding_dist_inference_backward", &Encoding_Dist_Inference_Backward_CUDA,
        "Assign Inference backward (CUDA)");
  m.def("aggregatev2_forward", &AggregateV2_Forward_CUDA, "AggregateV2 forward (CUDA)");
  m.def("aggregatev2_backward", &AggregateV2_Backward_CUDA, "AggregateV2 backward (CUDA)");
  m.def("leaky_relu_forward", &LeakyRelu_Forward_CUDA, "Learky ReLU forward (CUDA)");
  m.def("leaky_relu_backward", &LeakyRelu_Backward_CUDA, "Learky ReLU backward (CUDA)");
}
