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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class Decoder(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.squeeze = conv(in_channels, squeeze_channels)

    def forward(self, x, encoder_features):
        x = self.squeeze(x)
        x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
                          mode='bilinear', align_corners=True)
        up = torch.cat([encoder_features, x], 1)
        return up


class FOTSModel(nn.Module):
    def __init__(self, crop_height=640, args=None):
        super().__init__()
        self.crop_height = crop_height
        self.resnet = torchvision.models.resnet34(pretrained=False)
        import torch
        self.resnet.load_state_dict(torch.load(args.train_folder+'/resnet34-333f7ec4.pth'))
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            conv(512, 512, stride=2),
            conv(512, 1024)
        )

        self.decoder4 = Decoder(1024, 512)
        self.decoder3 = Decoder(1024, 256)
        self.decoder2 = Decoder(512, 128)
        self.decoder1 = Decoder(256, 64)
        self.remove_artifacts = conv(128, 64)

        self.confidence = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
        self.distances = conv(64, 4, kernel_size=1, padding=0, bn=False, relu=False)
        self.angle = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        f = self.center(e4)

        d4 = self.decoder4(f, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        final = self.remove_artifacts(d1)

        confidence = self.confidence(final)
        distances = self.distances(final)
        distances = torch.sigmoid(distances) * self.crop_height
        angle = self.angle(final)
        angle = torch.sigmoid(angle) * np.pi / 2

        return confidence, distances, angle


# class FOTSModel(nn.Module):
#     """This model is described in the paper, but it trains slower and gives slightly worse results"""
#     def __init__(self, crop_height=640):
#         super().__init__()
#         self.crop_height = crop_height
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         self.conv1 = nn.Sequential(
#             self.resnet.conv1,
#             self.resnet.bn1,
#             self.resnet.relu,
#         )  # 64 * 4
#         self.encoder1 = self.resnet.layer1  # 64 * 4
#         self.encoder2 = self.resnet.layer2  # 128 * 4
#         self.encoder3 = self.resnet.layer3  # 256 * 4
#         self.encoder4 = self.resnet.layer4  # 512 * 4

#         self.decoder3 = Decoder(512 * 4, 256 * 4)
#         self.decoder2 = Decoder(256 * 4 * 2, 128 * 4)
#         self.decoder1 = Decoder(128 * 4 * 2, 64 * 4)

#         self.confidence = conv(64 * 4 * 2, 1, kernel_size=1, padding=0, bn=False, relu=False)
#         self.distances = conv(64 * 4 * 2, 4, kernel_size=1, padding=0, bn=False, relu=False)
#         self.angle = conv(64 * 4 * 2, 1, kernel_size=1, padding=0, bn=False, relu=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)

#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)

#         d3 = self.decoder3(e4, e3)
#         d2 = self.decoder2(d3, e2)
#         d1 = self.decoder1(d2, e1)

#         confidence = self.confidence(d1)
#         distances = self.distances(d1)
#         distances = torch.sigmoid(distances) * self.crop_height
#         angle = self.angle(d1)
#         angle = torch.sigmoid(angle) * np.pi / 2

#         return confidence, distances, angle
