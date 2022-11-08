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
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16

from mmocr.models.builder import NECKS


@NECKS.register_module()
class FPNF(BaseModule):
    """FPN-like fusion module in Shape Robust Text Detection with Progressive
    Scale Expansion Network.

    Args:
        in_channels (list[int]): A list of number of input channels.
        out_channels (int): The number of output channels.
        fusion_type (str): Type of the final feature fusion layer. Available
            options are "concat" and "add".
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 fusion_type='concat',
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.backbone_end_level = len(in_channels)
        for i in range(self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

            if i < self.backbone_end_level - 1:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = 1024
        elif self.fusion_type == 'add':
            feature_channels = 256
        else:
            raise NotImplementedError

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    @auto_fp16()
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C_i, H_i, W_i)`. It usually expects 4 tensors
                (C2-C5 features) from ResNet.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            # step 2: smooth level i-1
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # upsample and cont
        bottom_shape = laterals[0].shape[2:]
        for i in range(1, used_backbone_levels):
            laterals[i] = F.interpolate(
                laterals[i], size=bottom_shape, mode='nearest')

        if self.fusion_type == 'concat':
            out = torch.cat(laterals, 1)
        elif self.fusion_type == 'add':
            out = laterals[0]
            for i in range(1, used_backbone_levels):
                out += laterals[i]
        else:
            raise NotImplementedError
        out = self.output_convs(out)

        return out
