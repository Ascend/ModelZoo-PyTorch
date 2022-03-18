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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

class FFDNet(nn.Module):

    def __init__(self, is_gray):
        super(FFDNet, self).__init__()

        if is_gray:
            self.num_conv_layers = 15 # all layers number
            self.downsampled_channels = 5 # Conv_Relu in
            self.num_feature_maps = 64 # Conv_Bn_Relu in
            self.output_features = 4 # Conv out
        else:
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.num_feature_maps = 96
            self.output_features = 12
            
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)

        x_up = utils.downsample(x.data) # 4 * C * H/2 * W/2
        x_up = x_up.npu()
        x_cat = torch.cat((noise_map.data, x_up.to(torch.float16)), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)

        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = utils.upsample(h_dncnn)
        return y_pred
