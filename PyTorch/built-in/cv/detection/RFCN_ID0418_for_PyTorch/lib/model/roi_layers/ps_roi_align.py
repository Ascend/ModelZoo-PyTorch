# BSD 3-Clause License
#
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

import math
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch
from model import _C


class _PSROIAlign(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_rois, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        ctx.spatial_scale = spatial_scale  # 1./16.
        ctx.roi_size = roi_size  # 7
        ctx.sampling_ratio = sampling_ratio  # 2
        ctx.pooled_dim = pooled_dim  # 10
        ctx.feature_size = bottom_data.size()  # (B, 490, H, W)
        num_rois = bottom_rois.size(0)  # B*K
        # (B*K, 10, 7, 7)
        top_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.float32).to(bottom_data.device)
        # (B*K, 10, 7, 7)
        argmax_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.int32).to(bottom_data.device)
        if bottom_data.is_cuda:
            _C.ps_roi_align_forward(bottom_data,    # (B, 490, H, W)
                                    bottom_rois,    # (B*K, 5), e.g. K = 128
                                    top_data,       # (B*K, 10, 7, 7)
                                    argmax_data,    # (B*K, 10, 7, 7)
                                    spatial_scale,  # 1./16.
                                    roi_size,       # 7
                                    sampling_ratio  # 2
                                    )
            ctx.save_for_backward(bottom_rois, argmax_data)
        else:
            raise NotImplementedError

        return top_data

    @staticmethod
    @once_differentiable
    def backward(ctx, top_diff):
        spatial_scale = ctx.spatial_scale  # 1./16.
        roi_size = ctx.roi_size  # 7
        sampling_ratio = ctx.sampling_ratio  # 2
        batch_size, channels, height, width = ctx.feature_size
        [bottom_rois, argmax_data] = ctx.saved_tensors
        bottom_diff = None
        if ctx.needs_input_grad[0]:
            bottom_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float32).to(top_diff.device)
            _C.ps_roi_align_backward(top_diff,      # (B*K, 10, 7, 7)
                                     argmax_data,   # (B*K, 10, 7, 7)
                                     bottom_rois,   # (B*K, 10, 7, 7)
                                     bottom_diff,   # (B, 490, H, W)
                                     spatial_scale, # 1./16.
                                     roi_size,      # 7
                                     sampling_ratio # 2
                                     )

        return bottom_diff, None, None, None, None, None


ps_roi_align = _PSROIAlign.apply


class PSROIAlign(nn.Module):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(PSROIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return ps_roi_align(bottom_data,  # (B, 490, H, W)
                            bottom_rois,  # (B*K, 5)
                            self.spatial_scale,  # 1./16.
                            self.roi_size,  # 7
                            self.sampling_ratio,  # 2
                            self.pooled_dim  # 10
                            )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", roi_size=" + str(self.roi_size)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", pooled_dim=" + str(self.pooled_dim)
        tmpstr += ")"
        return tmpstr
