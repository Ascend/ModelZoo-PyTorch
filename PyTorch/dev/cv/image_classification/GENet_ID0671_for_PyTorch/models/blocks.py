#BSD 3-Clause License
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
#Copyright (c) 2017, 
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#================================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=2, padding=1):
        super(Downblock, self).__init__()

        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x


class GEBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, spatial, extent=0, extra_params=True, mlp=True, dropRate=0.0):
        # If extent is zero, assuming global.

        super(GEBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

        self.extent = extent

        if extra_params is True:
            if extent == 0:
                # Global DW Conv + BN
                self.downop = Downblock(out_planes, relu=False, kernel_size=spatial, stride=1, padding=0)
            elif extent == 2:
                self.downop = Downblock(out_planes, relu=False)

            elif extent == 4:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))
            elif extent == 8:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))

            else:

                raise NotImplementedError('Extent must be 0,2,4 or 8 for now')

        else:
            if extent == 0:
                self.downop = nn.AdaptiveAvgPool2d(1)

            else:
                self.downop = nn.AdaptiveAvgPool2d(spatial // extent)

        if mlp is True:
            self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False),
                                     )
        else:
            self.mlp = Identity()

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        # Assuming squares because lazy.
        shape_in = out.shape[-1]

        # Down, up, sigmoid
        map = self.downop(out)
        map = self.mlp(map)
        map = F.interpolate(map, shape_in)
        map = torch.sigmoid(map)

        out = out * map

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
