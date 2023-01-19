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

#include "rotated_feature_align_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void rotated_feature_align_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto features = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  rotated_feature_align_forward(features, best_bboxes, output, spatial_scale,
                                points);
}

void rotated_feature_align_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  rotated_feature_align_backward(grad_output, best_bboxes, grad_input,
                                 spatial_scale, points);
}

void rotated_feature_align_forward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto features = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  rotated_feature_align_forward(features, best_bboxes, output, spatial_scale,
                                points);
}
#endif

void rotated_feature_align_backward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  rotated_feature_align_backward(grad_output, best_bboxes, grad_input,
                                 spatial_scale, points);
}

PARROTS_EXTENSION_REGISTER(rotated_feature_align_forward)
    .attr("spatial_scale")
    .attr("points")
    .input(2)
    .output(1)
    .apply(rotated_feature_align_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(rotated_feature_align_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(rotated_feature_align_backward)
    .attr("spatial_scale")
    .attr("points")
    .input(2)
    .output(1)
    .apply(rotated_feature_align_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(rotated_feature_align_backward_cuda_parrots)
#endif
    .done();
