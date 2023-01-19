# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "modulated_deform_conv_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void modulated_deform_conv_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);

  modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output,
                                columns, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dilation_h, dilation_w, group,
                                deformable_group, with_bias);
}

void modulated_deform_conv_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto columns = buildATensor(ctx, outs[0]);
  auto grad_input = buildATensor(ctx, outs[1]);
  auto grad_weight = buildATensor(ctx, outs[2]);
  auto grad_bias = buildATensor(ctx, outs[3]);
  auto grad_offset = buildATensor(ctx, outs[4]);
  auto grad_mask = buildATensor(ctx, outs[5]);
  auto grad_output = buildATensor(ctx, outs[6]);
  modulated_deform_conv_backward(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}
#endif

void modulated_deform_conv_forward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);

  modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output,
                                columns, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dilation_h, dilation_w, group,
                                deformable_group, with_bias);
}

void modulated_deform_conv_backward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto columns = buildATensor(ctx, outs[0]);
  auto grad_input = buildATensor(ctx, outs[1]);
  auto grad_weight = buildATensor(ctx, outs[2]);
  auto grad_bias = buildATensor(ctx, outs[3]);
  auto grad_offset = buildATensor(ctx, outs[4]);
  auto grad_mask = buildATensor(ctx, outs[5]);
  auto grad_output = buildATensor(ctx, outs[6]);
  modulated_deform_conv_backward(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}
PARROTS_EXTENSION_REGISTER(modulated_deform_conv_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(2)
    .apply(modulated_deform_conv_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(modulated_deform_conv_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_backward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(7)
    .apply(modulated_deform_conv_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(modulated_deform_conv_backward_cuda_parrots)
#endif
    .done();
