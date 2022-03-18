// Copyright 2021 Huawei Technologies Co., Ltd
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// modify from
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/SigmoidFocalLoss.h
#include <torch/extension.h>

at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                         const at::Tensor &targets,
                                         const int num_classes,
                                         const float gamma, const float alpha);

at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                          const at::Tensor &targets,
                                          const at::Tensor &d_losses,
                                          const int num_classes,
                                          const float gamma, const float alpha);

// Interface for Python
at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
                                    const at::Tensor &targets,
                                    const int num_classes, const float gamma,
                                    const float alpha) {
  if (logits.type().is_cuda()) {
    return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma,
                                         alpha);
  }
  AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
                                     const at::Tensor &targets,
                                     const at::Tensor &d_losses,
                                     const int num_classes, const float gamma,
                                     const float alpha) {
  if (logits.is_cuda()) {
    return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses,
                                          num_classes, gamma, alpha);
  }
  AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SigmoidFocalLoss_forward,
        "SigmoidFocalLoss forward (CUDA)");
  m.def("backward", &SigmoidFocalLoss_backward,
        "SigmoidFocalLoss backward (CUDA)");
}
