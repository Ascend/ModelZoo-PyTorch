
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS

from .builder import LINEAR_LAYERS


@LINEAR_LAYERS.register_module(name='NormedLinear')
class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
    """

    def __init__(self, *args, tempearture=20, power=1.0, eps=1e-6, **kwargs):
        super(NormedLinear, self).__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        weight_ = self.weight / (
            self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        return F.linear(x_, weight_, self.bias)


@CONV_LAYERS.register_module(name='NormedConv2d')
class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Default to False.
    """

    def __init__(self,
                 *args,
                 tempearture=20,
                 power=1.0,
                 eps=1e-6,
                 norm_over_kernel=False,
                 **kwargs):
        super(NormedConv2d, self).__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps

    def forward(self, x):
        if not self.norm_over_kernel:
            weight_ = self.weight / (
                self.weight.norm(dim=1, keepdim=True).pow(self.power) +
                self.eps)
        else:
            weight_ = self.weight / (
                self.weight.view(self.weight.size(0), -1).norm(
                    dim=1, keepdim=True).pow(self.power)[..., None, None] +
                self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        else:
            if torch.__version__ >= '1.8':
                x_ = self._conv_forward(x_, weight_, self.bias)
            else:
                x_ = self._conv_forward(x_, weight_)
        return x_
