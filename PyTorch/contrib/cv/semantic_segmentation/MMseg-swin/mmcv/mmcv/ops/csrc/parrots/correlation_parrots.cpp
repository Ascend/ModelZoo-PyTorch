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

#include "correlation_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void correlation_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto input1 = buildATensor(ctx, ins[0]);
  auto input2 = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);

  correlation_forward(input1, input2, output, kH, kW, patchH, patchW, padH,
                      padW, dilationH, dilationW, dilation_patchH,
                      dilation_patchW, dH, dW);
}

void correlation_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto input1 = buildATensor(ctx, ins[1]);
  auto input2 = buildATensor(ctx, ins[2]);

  auto grad_input1 = buildATensor(ctx, outs[0]);
  auto grad_input2 = buildATensor(ctx, outs[1]);

  correlation_backward(grad_output, input1, input2, grad_input1, grad_input2,
                       kH, kW, patchH, patchW, padH, padW, dilationH, dilationW,
                       dilation_patchH, dilation_patchW, dH, dW);
}
#endif

void correlation_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto input1 = buildATensor(ctx, ins[0]);
  auto input2 = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);

  correlation_forward(input1, input2, output, kH, kW, patchH, patchW, padH,
                      padW, dilationH, dilationW, dilation_patchH,
                      dilation_patchW, dH, dW);
}

void correlation_backward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto input1 = buildATensor(ctx, ins[1]);
  auto input2 = buildATensor(ctx, ins[2]);

  auto grad_input1 = buildATensor(ctx, outs[0]);
  auto grad_input2 = buildATensor(ctx, outs[1]);

  correlation_backward(grad_output, input1, input2, grad_input1, grad_input2,
                       kH, kW, patchH, patchW, padH, padW, dilationH, dilationW,
                       dilation_patchH, dilation_patchW, dH, dW);
}

PARROTS_EXTENSION_REGISTER(correlation_forward)
    .attr("kH")
    .attr("kW")
    .attr("patchH")
    .attr("patchW")
    .attr("padH")
    .attr("padW")
    .attr("dilationH")
    .attr("dilationW")
    .attr("dilation_patchH")
    .attr("dilation_patchW")
    .attr("dH")
    .attr("dW")
    .input(2)
    .output(1)
    .apply(correlation_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(correlation_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(correlation_backward)
    .attr("kH")
    .attr("kW")
    .attr("patchH")
    .attr("patchW")
    .attr("padH")
    .attr("padW")
    .attr("dilationH")
    .attr("dilationW")
    .attr("dilation_patchH")
    .attr("dilation_patchW")
    .attr("dH")
    .attr("dW")
    .input(3)
    .output(2)
    .apply(correlation_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(correlation_backward_cuda_parrots)
#endif
    .done();
