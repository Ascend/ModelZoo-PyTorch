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
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn

from mmocr.models.builder import HEADS


@HEADS.register_module()
class SegHead(BaseModule):
    """Head for segmentation based text recognition.

    Args:
        in_channels (int): Number of input channels :math:`C`.
        num_classes (int): Number of output classes :math:`C_{out}`.
        upsample_param (dict | None): Config dict for interpolation layer.
            Default: ``dict(scale_factor=1.0, mode='nearest')``
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=128,
                 num_classes=37,
                 upsample_param=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert upsample_param is None or isinstance(upsample_param, dict)

        self.upsample_param = upsample_param

        self.seg_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'))

        # prediction
        self.pred_conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, out_neck):
        """
        Args:
            out_neck (list[Tensor]): A list of tensor of shape
                :math:`(N, C_i, H_i, W_i)`. The network only uses the last one
                (``out_neck[-1]``).

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, kH, kW)` where
            :math:`k` is determined by ``upsample_param``.
        """

        seg_map = self.seg_conv(out_neck[-1])
        seg_map = self.pred_conv(seg_map)

        if self.upsample_param is not None:
            seg_map = F.interpolate(seg_map, **self.upsample_param)

        return seg_map
