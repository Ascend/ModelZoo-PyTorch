# Copyright 2020 Huawei Technologies Co., Ltd
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
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .acceleration import AcFusionLayer as Acceleration
from .forward_warp_gaussian import ForwardWarp as ForwardWarp
from .UNet2 import UNet2 as UNet
from .PWCNetnew import PWCNet


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False).npu()
    gridY = torch.tensor(gridY, requires_grad=False).npu()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2 * (x / W - 0.5)
    y = 2 * (y / H - 0.5)
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""

    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x


class QVI(nn.Module):
    """The quadratic model"""

    def __init__(self, path=""):
        super().__init__()
        self.flownet = PWCNet()
        self.acc = Acceleration()
        self.fwarp = ForwardWarp()  # 非常慢
        self.refinenet = UNet(20, 8)
        self.masknet = SmallMaskNet(38, 1)
        self.flownet.load_state_dict(torch.load(path, map_location="cpu"))

    def forward(self, I0, I1, I2, I3, t):
        if I0 is not None:
            F10 = self.flownet(I1, I0).float()
        else:
            F10 = None
        F12 = self.flownet(I1, I2).float()
        F21 = self.flownet(I2, I1).float()
        if I3 is not None:
            F23 = self.flownet(I2, I3).float()
        else:
            F23 = None
        if F10 is not None and F23 is not None:
            F1ta, F2ta = self.acc(F10, F12, F21, F23, t)
            F1t = F1ta
            F2t = F2ta
        else:
            F1t = t * F12
            F2t = (1 - t) * F21

        # Flow Reversal
        Ft1, norm1 = self.fwarp(F1t, F1t)
        Ft1 = -Ft1
        Ft2, norm2 = self.fwarp(F2t, F2t)
        Ft2 = -Ft2

        Ft1[norm1 > 0] = Ft1[norm1 > 0] / norm1[norm1 > 0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0] / norm2[norm2 > 0].clone()


        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)
        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, F12, F21, Ft1, Ft2], dim=1))
        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10 * torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10 * torch.tanh(output[:, 6:8])) + output[:, 2:4]
        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)
        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)
        It_warp = ((1 - t) * M * I1tf + t * (1 - M) * I2tf) / ((1 - t) * M + t * (1 - M)).clone()
        return It_warp
