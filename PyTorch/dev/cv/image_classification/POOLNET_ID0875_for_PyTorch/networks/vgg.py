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
import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [512, 512, 512]}
        self.extract = [8, 15, 22, 29] # [3, 8, 15, 22, 29]
        self.base = nn.ModuleList(vgg(self.cfg['tun'], 3))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.base.load_state_dict(model, strict=False)

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

class vgg16_locate(nn.Module):
    def __init__(self):
        super(vgg16_locate,self).__init__()
        self.vgg16 = vgg16()
        self.in_planes = 512
        self.out_planes = [512, 256, 128]

        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.vgg16(x)

        xls = [xs[-1]]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos
