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

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['rotated_feature_align_forward', 'rotated_feature_align_backward'])


class RotatedFeatureAlignFunction(Function):
    """Using the feature interpolation to obtain the position information
    correspond to the refined rotate anchors and reconstruct the feature maps
    in pixel-wise manner to achieve feature alignment.

    The details are described in the paper
    `R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating
    Object <https://arxiv.org/abs/1908.05612>`_.
    """

    @staticmethod
    def symbolic(g, features, best_rbboxes, spatial_scale, points):
        assert points in [1, 5]
        return g.op(
            'mmcv::MMCVRotatedFeatureAlign',
            features,
            best_rbboxes,
            spatial_scale_f=spatial_scale,
            points_i=points)

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, best_rbboxes: torch.Tensor,
                spatial_scale: float, points: int) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Input features with shape [N,C,H,W].
            best_rbboxes (torch.Tensor): Refined rotate anchors with
                shape [N,H,W,5]. Coordinate format (cx,cx,h,w,a).
            spatial_scale (float): The scale of feature map size and
                input image size.
            points (int, optional): The number of sample points.
                Only 1 and 5 are supported. Defaults to 1.

        Returns:
            torch.Tensor: Refined features with shape [N,C,H,W].
        """
        ctx.spatial_scale = spatial_scale
        ctx.points = points
        ctx.save_for_backward(best_rbboxes)
        assert points in [1, 5]
        output = torch.zeros_like(features)
        ext_module.rotated_feature_align_forward(
            features,
            best_rbboxes,
            output,
            spatial_scale=spatial_scale,
            points=points)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """
        Args:
            grad_output (torch.Tensor): The gradiant of output features
                with shape [N,C,H,W].

        Returns:
            torch.Tensor: The gradiant of input features with shape [N,C,H,W].
        """
        best_rbboxes = ctx.saved_tensors[0]
        points = ctx.points
        spatial_scale = ctx.spatial_scale
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(grad_output)
            ext_module.rotated_feature_align_backward(
                grad_output.contiguous(),
                best_rbboxes,
                grad_input,
                spatial_scale=spatial_scale,
                points=points)
        return grad_input, None, None, None


def rotated_feature_align(features: torch.Tensor,
                          best_rbboxes: torch.Tensor,
                          spatial_scale: float = 1 / 8,
                          points: int = 1) -> torch.Tensor:
    return RotatedFeatureAlignFunction.apply(features, best_rbboxes,
                                             spatial_scale, points)
