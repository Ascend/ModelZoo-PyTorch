# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import torch

import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import  _FCNHead
from ..config import cfg

__all__ = ["EDANet"]


@MODEL_REGISTRY.register()
class EDANet(SegBaseModel):
    def __init__(self):
        super(EDANet, self).__init__(need_backbone=False)

        self.layers = nn.ModuleList()
        self.dilation1 = [1, 1, 1, 2, 2]
        self.dilation2 = [2, 2, 4, 4, 8, 8, 16, 16]

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlock(3, 15))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlock(15, 60))

        # EDA module 1-1 ~ 1-5
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlock(260, 130))

        # EDA module 2-1 ~ 2-8
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))

        # Projection layer
        self.project_layer = nn.Conv2d(450, self.nclass, kernel_size=1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):

        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=True)
        # Bilinear interpolation x8
        # output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=True)
        #
        # # Bilinear interpolation x2 (inference only)
        # if not self.training:
        #     output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return tuple([output])


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)


class EDABlock(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()

        # k: growthrate
        # dropprob:a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated, 0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0, dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = torch.cat([output, input], 1)
        # print output.size() #check the output
        return output