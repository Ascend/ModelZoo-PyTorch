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

#include "assign_score_withk_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void assign_score_withk_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& points = buildATensor(ctx, ins[0]);
  const auto& centers = buildATensor(ctx, ins[1]);
  const auto& scores = buildATensor(ctx, ins[2]);
  const auto& knn_idx = buildATensor(ctx, ins[3]);

  auto output = buildATensor(ctx, outs[0]);
  assign_score_withk_forward(points, centers, scores, knn_idx, output, B, N0,
                             N1, M, K, O, aggregate);
}

void assign_score_withk_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& grad_out = buildATensor(ctx, ins[0]);
  const auto& points = buildATensor(ctx, ins[1]);
  const auto& centers = buildATensor(ctx, ins[2]);
  const auto& scores = buildATensor(ctx, ins[3]);
  const auto& knn_idx = buildATensor(ctx, ins[4]);

  auto grad_points = buildATensor(ctx, outs[0]);
  auto grad_centers = buildATensor(ctx, outs[1]);
  auto grad_scores = buildATensor(ctx, outs[2]);
  assign_score_withk_backward(grad_out, points, centers, scores, knn_idx,
                              grad_points, grad_centers, grad_scores, B, N0, N1,
                              M, K, O, aggregate);
}

PARROTS_EXTENSION_REGISTER(assign_score_withk_forward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(4)
    .output(1)
    .apply(assign_score_withk_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(assign_score_withk_backward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(5)
    .output(3)
    .apply(assign_score_withk_backward_cuda_parrots)
    .done();
#endif
