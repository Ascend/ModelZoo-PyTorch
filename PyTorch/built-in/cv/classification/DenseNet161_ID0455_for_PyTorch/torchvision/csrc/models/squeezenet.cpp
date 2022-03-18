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
*/#include "squeezenet.h"

#include <limits>
#include "modelsimpl.h"

namespace vision {
namespace models {
struct Fire : torch::nn::Module {
  torch::nn::Conv2d squeeze, expand1x1, expand3x3;

  Fire(
      int64_t inplanes,
      int64_t squeeze_planes,
      int64_t expand1x1_planes,
      int64_t expand3x3_planes)
      : squeeze(torch::nn::Conv2dOptions(inplanes, squeeze_planes, 1)),
        expand1x1(
            torch::nn::Conv2dOptions(squeeze_planes, expand1x1_planes, 1)),
        expand3x3(torch::nn::Conv2dOptions(squeeze_planes, expand3x3_planes, 3)
                      .padding(1)) {
    register_module("squeeze", squeeze);
    register_module("expand1x1", expand1x1);
    register_module("expand3x3", expand3x3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(squeeze->forward(x));
    return torch::cat(
        {torch::relu(expand1x1->forward(x)),
         torch::relu(expand3x3->forward(x))},
        1);
  }
};

SqueezeNetImpl::SqueezeNetImpl(double version, int64_t num_classes)
    : num_classes(num_classes) {
  if (modelsimpl::double_compare(version, 1.0)) {
    features = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, 7).stride(2)),
        torch::nn::Functional(modelsimpl::relu_),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(96, 16, 64, 64),
        Fire(128, 16, 64, 64),
        Fire(128, 32, 128, 128),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(256, 32, 128, 128),
        Fire(256, 48, 192, 192),
        Fire(384, 48, 192, 192),
        Fire(384, 64, 256, 256),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(512, 64, 256, 256));
  } else if (modelsimpl::double_compare(version, 1.1)) {
    features = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(2)),
        torch::nn::Functional(modelsimpl::relu_),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(64, 16, 64, 64),
        Fire(128, 16, 64, 64),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(128, 32, 128, 128),
        Fire(256, 32, 128, 128),
        torch::nn::Functional(torch::max_pool2d, 3, 2, 0, 1, true),
        Fire(256, 48, 192, 192),
        Fire(384, 48, 192, 192),
        Fire(384, 64, 256, 256),
        Fire(512, 64, 256, 256));
  } else
    TORCH_CHECK(
        false,
        "Unsupported SqueezeNet version ",
        version,
        ". 1_0 or 1_1 expected");

  // Final convolution is initialized differently from the rest
  auto final_conv =
      torch::nn::Conv2d(torch::nn::Conv2dOptions(512, num_classes, 1));

  classifier = torch::nn::Sequential(
      torch::nn::Dropout(0.5),
      final_conv,
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::adaptive_avg_pool2d, 1));

  register_module("features", features);
  register_module("classifier", classifier);

  for (auto& module : modules(/*include_self=*/false))
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
      if (M == final_conv.get())
        torch::nn::init::normal_(M->weight, 0.0, 0.01);
      else
        torch::nn::init::kaiming_uniform_(M->weight);

      if (M->options.with_bias())
        torch::nn::init::constant_(M->bias, 0);
    }
}

torch::Tensor SqueezeNetImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = classifier->forward(x);
  return x.view({x.size(0), -1});
}

SqueezeNet1_0Impl::SqueezeNet1_0Impl(int64_t num_classes)
    : SqueezeNetImpl(1.0, num_classes) {}

SqueezeNet1_1Impl::SqueezeNet1_1Impl(int64_t num_classes)
    : SqueezeNetImpl(1.1, num_classes) {}

} // namespace models
} // namespace vision
