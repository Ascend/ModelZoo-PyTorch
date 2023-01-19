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
import torch
import torch.nn as nn
from mmcv import is_tuple_of
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class LRASPPHead(BaseDecodeHead):
    """Lite R-ASPP (LRASPP) head is proposed in Searching for MobileNetV3.

    This head is the improved implementation of `Searching for MobileNetV3
    <https://ieeexplore.ieee.org/document/9008835>`_.

    Args:
        branch_channels (tuple[int]): The number of output channels in every
            each branch. Default: (32, 64).
    """

    def __init__(self, branch_channels=(32, 64), **kwargs):
        super(LRASPPHead, self).__init__(**kwargs)
        if self.input_transform != 'multiple_select':
            raise ValueError('in Lite R-ASPP (LRASPP) head, input_transform '
                             f'must be \'multiple_select\'. But received '
                             f'\'{self.input_transform}\'')
        assert is_tuple_of(branch_channels, int)
        assert len(branch_channels) == len(self.in_channels) - 1
        self.branch_channels = branch_channels

        self.convs = nn.Sequential()
        self.conv_ups = nn.Sequential()
        for i in range(len(branch_channels)):
            self.convs.add_module(
                f'conv{i}',
                nn.Conv2d(
                    self.in_channels[i], branch_channels[i], 1, bias=False))
            self.conv_ups.add_module(
                f'conv_up{i}',
                ConvModule(
                    self.channels + branch_channels[i],
                    self.channels,
                    1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=False))

        self.conv_up_input = nn.Conv2d(self.channels, self.channels, 1)

        self.aspp_conv = ConvModule(
            self.in_channels[-1],
            self.channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            bias=False)
        self.image_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=49, stride=(16, 20)),
            ConvModule(
                self.in_channels[2],
                self.channels,
                1,
                act_cfg=dict(type='Sigmoid'),
                bias=False))

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        x = inputs[-1]

        x = self.aspp_conv(x) * resize(
            self.image_pool(x),
            size=x.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.conv_up_input(x)

        for i in range(len(self.branch_channels) - 1, -1, -1):
            x = resize(
                x,
                size=inputs[i].size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            x = torch.cat([x, self.convs[i](inputs[i])], 1)
            x = self.conv_ups[i](x)

        return self.cls_seg(x)
