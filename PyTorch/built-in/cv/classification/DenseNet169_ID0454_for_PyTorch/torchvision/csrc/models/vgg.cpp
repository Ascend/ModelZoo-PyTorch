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
*/#include "vgg.h"

#include <unordered_map>
#include "modelsimpl.h"

namespace vision {
namespace models {
torch::nn::Sequential makeLayers(
    const std::vector<int>& cfg,
    bool batch_norm = false) {
  torch::nn::Sequential seq;
  auto channels = 3;

  for (const auto& V : cfg) {
    if (V <= -1)
      seq->push_back(torch::nn::Functional(modelsimpl::max_pool2d, 2, 2));
    else {
      seq->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(channels, V, 3).padding(1)));

      if (batch_norm)
        seq->push_back(torch::nn::BatchNorm(V));
      seq->push_back(torch::nn::Functional(modelsimpl::relu_));

      channels = V;
    }
  }

  return seq;
}

void VGGImpl::_initialize_weights() {
  for (auto& module : modules(/*include_self=*/false)) {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
      torch::nn::init::kaiming_normal_(
          M->weight,
          /*a=*/0,
          torch::nn::init::FanMode::FanOut,
          torch::nn::init::Nonlinearity::ReLU);
      torch::nn::init::constant_(M->bias, 0);
    } else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
      torch::nn::init::constant_(M->weight, 1);
      torch::nn::init::constant_(M->bias, 0);
    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
      torch::nn::init::normal_(M->weight, 0, 0.01);
      torch::nn::init::constant_(M->bias, 0);
    }
  }
}

VGGImpl::VGGImpl(
    torch::nn::Sequential features,
    int64_t num_classes,
    bool initialize_weights) {
  classifier = torch::nn::Sequential(
      torch::nn::Linear(512 * 7 * 7, 4096),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Dropout(),
      torch::nn::Linear(4096, 4096),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Dropout(),
      torch::nn::Linear(4096, num_classes));

  this->features = features;

  register_module("features", this->features);
  register_module("classifier", classifier);

  if (initialize_weights)
    _initialize_weights();
}

torch::Tensor VGGImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = torch::adaptive_avg_pool2d(x, {7, 7});
  x = x.view({x.size(0), -1});
  x = classifier->forward(x);
  return x;
}

// clang-format off
static std::unordered_map<char, std::vector<int>> cfgs = {
  {'A', {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
  {'B', {64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
  {'D', {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
  {'E', {64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1, 512, 512, 512, 512, -1,  512, 512, 512, 512, -1}}};
// clang-format on

VGG11Impl::VGG11Impl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['A']), num_classes, initialize_weights) {}

VGG13Impl::VGG13Impl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['B']), num_classes, initialize_weights) {}

VGG16Impl::VGG16Impl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['D']), num_classes, initialize_weights) {}

VGG19Impl::VGG19Impl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['E']), num_classes, initialize_weights) {}

VGG11BNImpl::VGG11BNImpl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['A'], true), num_classes, initialize_weights) {}

VGG13BNImpl::VGG13BNImpl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['B'], true), num_classes, initialize_weights) {}

VGG16BNImpl::VGG16BNImpl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['D'], true), num_classes, initialize_weights) {}

VGG19BNImpl::VGG19BNImpl(int64_t num_classes, bool initialize_weights)
    : VGGImpl(makeLayers(cfgs['E'], true), num_classes, initialize_weights) {}

} // namespace models
} // namespace vision
