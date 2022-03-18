/*
* BSD 3-Clause License
*
* Copyright (c) 2017 xxxx
* All rights reserved.
* Copyright 2021 Huawei Technologies Co., Ltd
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ============================================================================
*/#include "alexnet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
AlexNetImpl::AlexNetImpl(int64_t num_classes) {
  features = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2));

  classifier = torch::nn::Sequential(
      torch::nn::Dropout(),
      torch::nn::Linear(256 * 6 * 6, 4096),
      torch::nn::Functional(torch::relu),
      torch::nn::Dropout(),
      torch::nn::Linear(4096, 4096),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(4096, num_classes));

  register_module("features", features);
  register_module("classifier", classifier);
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = torch::adaptive_avg_pool2d(x, {6, 6});
  x = x.view({x.size(0), -1});
  x = classifier->forward(x);

  return x;
}

} // namespace models
} // namespace vision
