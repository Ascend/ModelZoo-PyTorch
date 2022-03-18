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
"""senet in pytorch



[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.npu
import os
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class BasicResidualSEBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)

def seresnet18():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2])

def seresnet34():
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3])

def seresnet50():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])

def seresnet101():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3])

def seresnet152():
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3])
