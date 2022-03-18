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
from model.block import *
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


class ESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=4, n_basic_block=23):
        super(ESRGAN, self).__init__()

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, 3), nn.ReLU())

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels, 3), nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
