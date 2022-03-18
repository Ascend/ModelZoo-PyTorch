// Copyright 2021 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void CARAFENAIVEForwardCUDAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor);

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor);

void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
}

void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
}
#endif

void carafe_naive_forward(Tensor features, Tensor masks, Tensor output,
                          int kernel_size, int group_size, int scale_factor) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(masks);
    CHECK_CUDA_INPUT(output);
    carafe_naive_forward_cuda(features, masks, output, kernel_size, group_size,
                              scale_factor);
#else
    AT_ERROR("CarafeNaive is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CarafeNaive is not implemented on CPU");
  }
}

void carafe_naive_backward(Tensor top_grad, Tensor features, Tensor masks,
                           Tensor bottom_grad, Tensor mask_grad,
                           int kernel_size, int group_size, int scale_factor) {
  if (top_grad.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(top_grad);
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(masks);
    CHECK_CUDA_INPUT(bottom_grad);
    CHECK_CUDA_INPUT(mask_grad);
    carafe_naive_backward_cuda(top_grad, features, masks, bottom_grad,
                               mask_grad, kernel_size, group_size,
                               scale_factor);
#else
    AT_ERROR("CarafeNaive is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CarafeNaive is not implemented on CPU");
  }
}
