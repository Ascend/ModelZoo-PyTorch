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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .stripe import *
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

__all__ = ['ResNet56']


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = FilterStripe(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = BatchNorm(planes)
        self.conv2 = FilterStripe(planes, out_planes, kernel_size=3, stride=1)
        self.bn2 = BatchNorm(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                FilterStripe(in_planes, out_planes, kernel_size=1, stride=stride),
                BatchNorm(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet56(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(ResNet56, self).__init__()
        if cfg is None:
            cfg = [16] * 9 + [32] * 9 + [64] * 9
        self.in_planes = 16
        self.conv1 = FilterStripe(3, 16)
        self.bn1 = BatchNorm(16)
        self.layer1 = self._make_layer(16, cfg[:9], 9, stride=1)
        self.layer2 = self._make_layer(32, cfg[9:18], 9, stride=2)
        self.layer3 = self._make_layer(64, cfg[18:], 9, stride=2)
        self.fc = Linear(64, num_classes)

    def _make_layer(self, out_planes, cfg, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(BasicBlock(self.in_planes, cfg[i], out_planes, stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def update_skeleton(self, sr, threshold):
        for key, m in self.named_modules():
            if isinstance(m, FilterStripe) and 'downsample' not in key:
                out_mask = m.update_skeleton(sr, threshold)
            elif 'layer' in key and 'bn1' in key:
                m.update_mask(out_mask)

    def prune(self, threshold):
        mask_num = 0
        for key, m in self.named_modules():
            if key.startswith('conv'):
                m._break(threshold)
            elif isinstance(m, BasicBlock):
                out_mask = m.conv1.prune_out(threshold)
                m.bn1.prune(out_mask)
                m.conv2.prune_in(out_mask)
                m.conv1._break(threshold)
                m.conv2._break(threshold)
            if isinstance(m, FilterStripe):
                mask_num += m.FilterSkeleton.nelement()
        return mask_num
