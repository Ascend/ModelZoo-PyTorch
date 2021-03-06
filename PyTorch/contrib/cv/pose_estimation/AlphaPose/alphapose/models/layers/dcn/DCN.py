# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
# -----------------------------------------------------
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
# ===========================================================================
import torch.nn as nn

from . import DeformConv, ModulatedDeformConv


class DCN(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes, dcn,
                 kernel_size, stride=1,
                 padding=0, bias=False):
        super(DCN, self).__init__()
        fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
        self.with_modulated_dcn = dcn.get('MODULATED', False)
        if fallback_on_stride:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=bias)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27

            self.conv_offset = nn.Conv2d(
                inplanes,
                self.deformable_groups * offset_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
            self.conv = conv_op(
                inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                deformable_groups=self.deformable_groups,
                bias=bias)

    def forward(self, x):
        if self.with_modulated_dcn:
            offset_mask = self.conv_offset(x)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv(x, offset, mask)
        else:
            offset = self.conv_offset(x)
            out = self.conv(x, offset)

        return out
