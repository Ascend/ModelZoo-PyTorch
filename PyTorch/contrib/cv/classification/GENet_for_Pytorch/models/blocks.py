# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
