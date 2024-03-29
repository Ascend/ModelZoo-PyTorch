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

#include "deform_conv_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void deform_conv_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& offset = buildATensor(ctx, ins[2]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);

  deform_conv_forward(input, weight, offset, output, columns, ones, kW, kH, dW,
                      dH, padW, padH, dilationW, dilationH, group,
                      deformable_group, im2col_step);
}

void deform_conv_backward_input_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradInput = buildATensor(ctx, outs[0]);
  auto gradOffset = buildATensor(ctx, outs[1]);
  auto weight = buildATensor(ctx, outs[2]);
  auto columns = buildATensor(ctx, outs[3]);

  deform_conv_backward_input(input, offset, gradOutput, gradInput, gradOffset,
                             weight, columns, kW, kH, dW, dH, padW, padH,
                             dilationW, dilationH, group, deformable_group,
                             im2col_step);
}

void deform_conv_backward_parameters_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  float scale;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<float>("scale", scale)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradWeight = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);
  deform_conv_backward_parameters(input, offset, gradOutput, gradWeight,
                                  columns, ones, kW, kH, dW, dH, padW, padH,
                                  dilationW, dilationH, group, deformable_group,
                                  scale, im2col_step);
}
#endif

void deform_conv_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& offset = buildATensor(ctx, ins[2]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);

  deform_conv_forward(input, weight, offset, output, columns, ones, kW, kH, dW,
                      dH, padW, padH, dilationW, dilationH, group,
                      deformable_group, im2col_step);
}

void deform_conv_backward_input_cpu_parrots(HostContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradInput = buildATensor(ctx, outs[0]);
  auto gradOffset = buildATensor(ctx, outs[1]);
  auto weight = buildATensor(ctx, outs[2]);
  auto columns = buildATensor(ctx, outs[3]);

  deform_conv_backward_input(input, offset, gradOutput, gradInput, gradOffset,
                             weight, columns, kW, kH, dW, dH, padW, padH,
                             dilationW, dilationH, group, deformable_group,
                             im2col_step);
}

void deform_conv_backward_parameters_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  float scale;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<float>("scale", scale)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradWeight = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);
  deform_conv_backward_parameters(input, offset, gradOutput, gradWeight,
                                  columns, ones, kW, kH, dW, dH, padW, padH,
                                  dilationW, dilationH, group, deformable_group,
                                  scale, im2col_step);
}

PARROTS_EXTENSION_REGISTER(deform_conv_forward)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("im2col_step")
    .input(3)
    .output(3)
    .apply(deform_conv_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(deform_conv_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_input)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("im2col_step")
    .input(3)
    .output(4)
    .apply(deform_conv_backward_input_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(deform_conv_backward_input_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_parameters)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("scale")
    .attr("im2col_step")
    .input(3)
    .output(3)
    .apply(deform_conv_backward_parameters_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(deform_conv_backward_parameters_cuda_parrots)
#endif
    .done();
