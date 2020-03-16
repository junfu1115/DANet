#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#ifdef _OPENMP
#include <omp.h>
#endif

template<typename scalar>
inline scalar IoU(scalar* rawInput, int idx_x, int idx_y) {
    scalar lr = std::fmin(rawInput[idx_x*4] + rawInput[idx_x*4+2],
                         rawInput[idx_y*4] + rawInput[idx_y*4+2]);
    scalar rl = std::fmax(rawInput[idx_x*4], rawInput[idx_y*4]);
    scalar tb = std::fmin(rawInput[idx_x*4+1] + rawInput[idx_x*4+3],
                         rawInput[idx_y*4+1] + rawInput[idx_y*4+3]);
    scalar bt = std::fmax(rawInput[idx_x*4+1], rawInput[idx_y*4+1]);
    scalar inter = std::fmax(0, lr-rl)*std::fmax(0, tb-bt);
    scalar uni = (rawInput[idx_x*4+2]*rawInput[idx_x*4+3] 
                 + rawInput[idx_y*4+2]*rawInput[idx_y*4+3] - inter);
    return inter/uni;
}


std::vector<at::Tensor> Non_Max_Suppression_CPU(
    const at::Tensor& input,
    const at::Tensor& scores,
    double thresh) {
  AT_ASSERT(input.ndimension() == 3);
  AT_ASSERT(scores.ndimension() == 2);
  AT_ASSERT(input.size(0) == scores.size(0));
  AT_ASSERT(input.size(1) == scores.size(1));
  AT_ASSERT(input.size(2) == 4);
  AT_ASSERT(input.is_contiguous());
  AT_ASSERT(scores.is_contiguous());
  AT_ASSERT(input.type().scalarType() == at::kFloat || input.type().scalarType() == at::kDouble);
  AT_ASSERT(scores.type().scalarType() == at::kFloat || scores.type().scalarType() == at::kDouble);
  AT_ASSERT(input.is_contiguous());
  AT_ASSERT(scores.is_contiguous());

 
  at::Tensor sorted_inds = std::get<1>(scores.sort(-1, true));
  //at::Tensor rawIdx = std::get<1>(scores.sort(-1, true));
  
  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = torch::zeros({batch_size, num_boxes}, input.type().toScalarType(at::kByte));
  //auto mask = input.type().toScalarType(at::kByte).tensor({batch_size, num_boxes});
  mask.fill_(1);
  auto *rawMask = mask.data<unsigned char>();
  auto *rawIdx = sorted_inds.data<int64_t>();

  if (input.type().scalarType() == at::kFloat)
  {
    auto *rawInput = input.data<float>();

    for(int batch=0; batch<batch_size; ++batch)
    {
      int pos=batch*num_boxes;
      while(pos < (1+batch)*num_boxes-1)
      {
#pragma omp parallel for
        for(int i=pos+1; i<num_boxes*(1+batch); ++i)
        {
          int idx_x = rawIdx[pos]+num_boxes*batch;
          int idx_y = rawIdx[i]+num_boxes*batch;
          if (IoU(rawInput, idx_x, idx_y) > thresh)
            rawMask[i] = 0;
        }
        ++pos;
        while(pos < (1+batch)*num_boxes-1 and (rawMask[pos] == 0))
          ++pos;
      }
    }
  }
  else
  {
    auto *rawInput = input.data<double>();
    for(int batch=0; batch<batch_size; ++batch)
    {
      int pos=batch*num_boxes;
      while(pos < (1+batch)*num_boxes-1)
      {
#pragma omp parallel for
        for(int i=pos+1; i<num_boxes*(1+batch); ++i)
        {
          int idx_x = rawIdx[pos]+num_boxes*batch;
          int idx_y = rawIdx[i]+num_boxes*batch;
          if (IoU(rawInput, idx_x, idx_y) > thresh)
            rawMask[i] = 0;
        }
        ++pos;
        while(pos < (1+batch)*num_boxes-1 and (rawMask[pos] == 0))
          ++pos;
      }
    }
  }
  //see ./cuda/NonMaxSuppression.cu for comment about return value.
  return {mask, sorted_inds};
}
