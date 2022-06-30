#!/usr/bin/env python
# coding: utf-8
#
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
#
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeplabv3 import _ASPP
from .resnet import _ConvBnReLU, _ResLayer, _Stem
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3Plus, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[5], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))

        # Decoder
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(304, 256, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(256, 256, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                ]
            )
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h_ = self.reduce(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return h


if __name__ == "__main__":
    model = DeepLabV3Plus(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
