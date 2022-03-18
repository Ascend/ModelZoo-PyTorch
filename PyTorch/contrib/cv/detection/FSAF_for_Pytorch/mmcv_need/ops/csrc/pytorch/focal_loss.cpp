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
void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
}

void softmax_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SoftmaxFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void softmax_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  SoftmaxFocalLossBackwardCUDAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha);
}
#endif

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);

    sigmoid_focal_loss_forward_cuda(input, target, weight, output, gamma,
                                    alpha);
#else
    AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SigmoidFocalLoss is not implemented on CPU");
  }
}

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(grad_input);

    sigmoid_focal_loss_backward_cuda(input, target, weight, grad_input, gamma,
                                     alpha);
#else
    AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SigmoidFocalLoss is not implemented on CPU");
  }
}

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);

    softmax_focal_loss_forward_cuda(input, target, weight, output, gamma,
                                    alpha);
#else
    AT_ERROR("SoftmaxFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SoftmaxFocalLoss is not implemented on CPU");
  }
}

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(buff);
    CHECK_CUDA_INPUT(grad_input);

    softmax_focal_loss_backward_cuda(input, target, weight, buff, grad_input,
                                     gamma, alpha);
#else
    AT_ERROR("SoftmaxFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SoftmaxFocalLoss is not implemented on CPU");
  }
}
