# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CONV_LAYERS


def conv_ws_2d(input: torch.Tensor,
               weight: torch.Tensor,
               bias: Optional[torch.Tensor] = None,
               stride: Union[int, Tuple[int, int]] = 1,
               padding: Union[int, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               groups: int = 1,
               eps: float = 1e-5) -> torch.Tensor:
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@CONV_LAYERS.register_module('ConvWS')
class ConvWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 eps: float = 1e-5):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


@CONV_LAYERS.register_module(name='ConvAWS')
class ConvAWS2d(nn.Conv2d):
    """AWS (Adaptive Weight Standardization)

    This is a variant of Weight Standardization
    (https://arxiv.org/pdf/1903.10520.pdf)
    It is used in DetectoRS to avoid NaN
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: True
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma',
                             torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta',
                             torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight: torch.Tensor) -> torch.Tensor:
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = (weight - mean) / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._get_weight(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def _load_from_state_dict(self, state_dict: OrderedDict, prefix: str,
                              local_metadata: Dict, strict: bool,
                              missing_keys: List[str],
                              unexpected_keys: List[str],
                              error_msgs: List[str]) -> None:
        """Override default load function.

        AWS overrides the function _load_from_state_dict to recover
        weight_gamma and weight_beta if they are missing. If weight_gamma and
        weight_beta are found in the checkpoint, this function will return
        after super()._load_from_state_dict. Otherwise, it will compute the
        mean and std of the pretrained weights and store them in weight_beta
        and weight_gamma.
        """

        self.weight_gamma.data.fill_(-1)
        local_missing_keys: List = []
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, local_missing_keys,
                                      unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            for k in local_missing_keys:
                missing_keys.append(k)
            return
        weight = self.weight.data
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_beta.data.copy_(mean)
        self.weight_gamma.data.copy_(std)
        missing_gamma_beta = [
            k for k in local_missing_keys
            if k.endswith('weight_gamma') or k.endswith('weight_beta')
        ]
        for k in missing_gamma_beta:
            local_missing_keys.remove(k)
        for k in local_missing_keys:
            missing_keys.append(k)
