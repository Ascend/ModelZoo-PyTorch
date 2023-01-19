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

from typing import Any, Tuple

import torch
from torch import nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['roipoint_pool3d_forward'])


class RoIPointPool3d(nn.Module):
    """Encode the geometry-specific features of each 3D proposal.

    Please refer to `Paper of PartA2 <https://arxiv.org/pdf/1907.03670.pdf>`_
    for more details.

    Args:
        num_sampled_points (int, optional): Number of samples in each roi.
            Default: 512.
    """

    def __init__(self, num_sampled_points: int = 512):
        super().__init__()
        self.num_sampled_points = num_sampled_points

    def forward(self, points: torch.Tensor, point_features: torch.Tensor,
                boxes3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, C).
            point_features (torch.Tensor): Features of input points whose shape
                is (B, N, C).
            boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the pooled features whose shape is (B, M, 512, 3 + C). The
            second is an empty flag whose shape is (B, M).
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d,
                                            self.num_sampled_points)


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            point_features: torch.Tensor,
            boxes3d: torch.Tensor,
            num_sampled_points: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, C).
            point_features (torch.Tensor): Features of input points whose shape
                is (B, N, C).
            boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).
            num_sampled_points (int, optional): The num of sampled points.
                Default: 512.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the pooled features whose shape is (B, M, 512, 3 + C). The
            second is an empty flag whose shape is (B, M).
        """
        assert len(points.shape) == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[
            1], point_features.shape[2]
        pooled_boxes3d = boxes3d.view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros(
            (batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros(
            (batch_size, boxes_num)).int()

        ext_module.roipoint_pool3d_forward(points.contiguous(),
                                           pooled_boxes3d.contiguous(),
                                           point_features.contiguous(),
                                           pooled_features, pooled_empty_flag)

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
