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

#include "carafe_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
/*
 * void carafe_forward_cuda(Tensor features, Tensor masks, Tensor rfeatures,
 *                          Tensor routput, Tensor rmasks, Tensor output,
 *                          int kernel_size, int group_size, int scale_factor);
 */
void carafe_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& features = buildATensor(ctx, ins[0]);
  const auto& masks = buildATensor(ctx, ins[1]);

  auto rfeatures = buildATensor(ctx, outs[0]);
  auto routput = buildATensor(ctx, outs[1]);
  auto rmasks = buildATensor(ctx, outs[2]);
  auto output = buildATensor(ctx, outs[3]);

  carafe_forward_cuda(features, masks, rfeatures, routput, rmasks, output,
                      kernel_size, group_size, scale_factor);
}

/*
 * void carafe_backward_cuda(Tensor top_grad, Tensor rfeatures, Tensor masks,
 *                           Tensor rtop_grad, Tensor rbottom_grad_hs,
 *                           Tensor rbottom_grad, Tensor rmask_grad,
 *                           Tensor bottom_grad, Tensor mask_grad, int
 * kernel_size, int group_size, int scale_factor);
 */
void carafe_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                  const OperatorBase::in_list_t& ins,
                                  OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& top_grad = buildATensor(ctx, ins[0]);
  const auto& rfeatures = buildATensor(ctx, ins[1]);
  const auto& masks = buildATensor(ctx, ins[2]);

  auto rtop_grad = buildATensor(ctx, outs[0]);
  auto rbottom_grad_hs = buildATensor(ctx, outs[1]);
  auto rbottom_grad = buildATensor(ctx, outs[2]);
  auto rmask_grad = buildATensor(ctx, outs[3]);
  auto bottom_grad = buildATensor(ctx, outs[4]);
  auto mask_grad = buildATensor(ctx, outs[5]);

  carafe_backward_cuda(top_grad, rfeatures, masks, rtop_grad, rbottom_grad_hs,
                       rbottom_grad, rmask_grad, bottom_grad, mask_grad,
                       kernel_size, group_size, scale_factor);
}

PARROTS_EXTENSION_REGISTER(carafe_forward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(2)
    .output(4)
    .apply(carafe_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(carafe_backward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(3)
    .output(6)
    .apply(carafe_backward_cuda_parrots)
    .done();
#endif
