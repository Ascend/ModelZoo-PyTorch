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
# limitations under the License.

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial

#
# Helper modules
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):
        
        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B,K,H,W = x.size()
        x = x.view(B*K,1,H,W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x.cpu(), [d]*4, mode='replicate')
            x_pad = x_pad.npu()
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B,K,-1,H,W)

class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)

        return x.std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)

#
# PAMR module
#
class PAMR(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(PAMR, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B,K,H,W = x.size()
        _,C,_,_ = mask.size()

        x_std = self.aff_std(x)

        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)

        # xvals: [BxCxHxW]
        return mask
