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
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class NRTRModalityTransform(BaseModule):

    def __init__(self,
                 input_channels=3,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_1 = nn.ReLU(True)
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_2 = nn.ReLU(True)
        self.bn_2 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        n, c, h, w = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(n, w, h * c)

        x = self.linear(x)

        x = x.permute(0, 2, 1).contiguous().view(n, -1, 1, w)

        return x
