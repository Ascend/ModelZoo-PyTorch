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
*/#include "resnet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
namespace _resnetimpl {
torch::nn::Conv2d conv3x3(
    int64_t in,
    int64_t out,
    int64_t stride,
    int64_t groups) {
  torch::nn::Conv2dOptions O(in, out, 3);
  O.padding(1).stride(stride).groups(groups).with_bias(false);
  return torch::nn::Conv2d(O);
}

torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride) {
  torch::nn::Conv2dOptions O(in, out, 1);
  O.stride(stride).with_bias(false);
  return torch::nn::Conv2d(O);
}

int BasicBlock::expansion = 1;
int Bottleneck::expansion = 4;

BasicBlock::BasicBlock(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    torch::nn::Sequential downsample,
    int64_t groups,
    int64_t base_width)
    : stride(stride), downsample(downsample) {
  TORCH_CHECK(
      groups == 1 && base_width == 64,
      "BasicBlock only supports groups=1 and base_width=64");

  // Both conv1 and downsample layers downsample the input when stride != 1
  conv1 = conv3x3(inplanes, planes, stride);
  conv2 = conv3x3(planes, planes);

  bn1 = torch::nn::BatchNorm(planes);
  bn2 = torch::nn::BatchNorm(planes);

  register_module("conv1", conv1);
  register_module("conv2", conv2);

  register_module("bn1", bn1);
  register_module("bn2", bn2);

  if (!downsample.is_empty())
    register_module("downsample", this->downsample);
}

Bottleneck::Bottleneck(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    torch::nn::Sequential downsample,
    int64_t groups,
    int64_t base_width)
    : stride(stride), downsample(downsample) {
  auto width = int64_t(planes * (base_width / 64.)) * groups;

  // Both conv2 and downsample layers downsample the input when stride != 1
  conv1 = conv1x1(inplanes, width);
  conv2 = conv3x3(width, width, stride, groups);
  conv3 = conv1x1(width, planes * expansion);

  bn1 = torch::nn::BatchNorm(width);
  bn2 = torch::nn::BatchNorm(width);
  bn3 = torch::nn::BatchNorm(planes * expansion);

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("conv3", conv3);

  register_module("bn1", bn1);
  register_module("bn2", bn2);
  register_module("bn3", bn3);

  if (!downsample.is_empty())
    register_module("downsample", this->downsample);
}

torch::Tensor Bottleneck::forward(torch::Tensor X) {
  auto identity = X;

  auto out = conv1->forward(X);
  out = bn1->forward(out).relu_();

  out = conv2->forward(out);
  out = bn2->forward(out).relu_();

  out = conv3->forward(out);
  out = bn3->forward(out);

  if (!downsample.is_empty())
    identity = downsample->forward(X);

  out += identity;
  return out.relu_();
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
  auto identity = x;

  auto out = conv1->forward(x);
  out = bn1->forward(out).relu_();

  out = conv2->forward(out);
  out = bn2->forward(out);

  if (!downsample.is_empty())
    identity = downsample->forward(x);

  out += identity;
  return out.relu_();
}
} // namespace _resnetimpl

ResNet18Impl::ResNet18Impl(int64_t num_classes, bool zero_init_residual)
    : ResNetImpl({2, 2, 2, 2}, num_classes, zero_init_residual) {}

ResNet34Impl::ResNet34Impl(int64_t num_classes, bool zero_init_residual)
    : ResNetImpl({3, 4, 6, 3}, num_classes, zero_init_residual) {}

ResNet50Impl::ResNet50Impl(int64_t num_classes, bool zero_init_residual)
    : ResNetImpl({3, 4, 6, 3}, num_classes, zero_init_residual) {}

ResNet101Impl::ResNet101Impl(int64_t num_classes, bool zero_init_residual)
    : ResNetImpl({3, 4, 23, 3}, num_classes, zero_init_residual) {}

ResNet152Impl::ResNet152Impl(int64_t num_classes, bool zero_init_residual)
    : ResNetImpl({3, 8, 36, 3}, num_classes, zero_init_residual) {}

ResNext50_32x4dImpl::ResNext50_32x4dImpl(
    int64_t num_classes,
    bool zero_init_residual)
    : ResNetImpl({3, 4, 6, 3}, num_classes, zero_init_residual, 32, 4) {}

ResNext101_32x8dImpl::ResNext101_32x8dImpl(
    int64_t num_classes,
    bool zero_init_residual)
    : ResNetImpl({3, 4, 23, 3}, num_classes, zero_init_residual, 32, 8) {}

WideResNet50_2Impl::WideResNet50_2Impl(
    int64_t num_classes,
    bool zero_init_residual)
    : ResNetImpl({3, 4, 6, 3}, num_classes, zero_init_residual, 1, 64 * 2) {}

WideResNet101_2Impl::WideResNet101_2Impl(
    int64_t num_classes,
    bool zero_init_residual)
    : ResNetImpl({3, 4, 23, 3}, num_classes, zero_init_residual, 1, 64 * 2) {}

} // namespace models
} // namespace vision
