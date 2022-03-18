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
void TINShiftForwardCUDAKernelLauncher(Tensor input, Tensor shift,
                                       Tensor output);

void TINShiftBackwardCUDAKernelLauncher(Tensor grad_output, Tensor shift,
                                        Tensor grad_input);

void tin_shift_forward_cuda(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardCUDAKernelLauncher(input, shift, output);
}

void tin_shift_backward_cuda(Tensor grad_output, Tensor shift,
                             Tensor grad_input) {
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input);
}

#endif

void tin_shift_forward(Tensor input, Tensor shift, Tensor output) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(shift);
    CHECK_CUDA_INPUT(output);

    tin_shift_forward_cuda(input, shift, output);
#else
    AT_ERROR("TINShift is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("TINShift is not implemented on CPU");
  }
}

void tin_shift_backward(Tensor grad_output, Tensor shift, Tensor grad_input) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(shift);
    CHECK_CUDA_INPUT(grad_input);

    tin_shift_backward_cuda(grad_output, shift, grad_input);
#else
    AT_ERROR("TINShift is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("TINShift is not implemented on CPU");
  }
}
