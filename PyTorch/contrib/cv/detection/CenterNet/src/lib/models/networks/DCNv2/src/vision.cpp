/*
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
*/
#include "dcn_v2.h"
#include <torch/script.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward,
        "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward,
        "dcn_v2_psroi_pooling_backward");
}

inline at::Tensor dcn_v2(const at::Tensor &input, const at::Tensor &weight,
                         const at::Tensor &bias, const at::Tensor &offset,
                         const at::Tensor &mask, const int64_t kernel_h,
                         const int64_t kernel_w, const int64_t stride_h,
                         const int64_t stride_w, const int64_t pad_h,
                         const int64_t pad_w, const int64_t dilation_h,
                         const int64_t dilation_w,
                         const int64_t deformable_group) {
  return dcn_v2_forward(input, weight, bias, offset, mask, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dilation_h,
                        dilation_w, deformable_group);
}
static auto registry = torch::RegisterOperators().op("custom::dcn_v2", &dcn_v2);
