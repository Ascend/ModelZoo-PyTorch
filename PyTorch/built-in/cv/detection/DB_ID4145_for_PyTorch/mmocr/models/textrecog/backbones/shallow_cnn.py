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
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class ShallowCNN(BaseModule):
    """Implement Shallow CNN block for SATRN.

    SATRN: `On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    <https://arxiv.org/pdf/1910.04396.pdf>`_.

    Args:
        base_channels (int): Number of channels of input image tensor
            :math:`D_i`.
        hidden_dim (int): Size of hidden layers of the model :math:`D_m`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 input_channels=1,
                 hidden_dim=512,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(input_channels, int)
        assert isinstance(hidden_dim, int)

        self.conv1 = ConvModule(
            input_channels,
            hidden_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            hidden_dim // 2,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image feature :math:`(N, D_i, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m, H/4, W/4)`.
        """

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x
