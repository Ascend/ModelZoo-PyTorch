#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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
"""
Created on Sat Jun  8 15:45:16 2019

@author: viswanatha
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch
from mobilenet_ssd_priors import *
from MobileNetV2 import MobileNetV2, MobileNetV2_pretrained
if torch.__version__ >= "1.8":
    import torch_npu
import torch.npu
import os


NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, backbone_net):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4,
        }
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        if backbone_net == "MobileNetV2":
            self.loc_conv4_3 = nn.Conv2d(
                96, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv7 = nn.Conv2d(
                1280, n_boxes["conv7"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv8_2 = nn.Conv2d(
                512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv9_2 = nn.Conv2d(
                256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv10_2 = nn.Conv2d(
                256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv11_2 = nn.Conv2d(
                256, n_boxes["conv11_2"] * 4, kernel_size=3, padding=1
            )

            # Class prediction convolutions (predict classes in localization boxes)
            self.cl_conv4_3 = nn.Conv2d(
                96, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv7 = nn.Conv2d(
                1280, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv8_2 = nn.Conv2d(
                512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv9_2 = nn.Conv2d(
                256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv10_2 = nn.Conv2d(
                256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv11_2 = nn.Conv2d(
                256, n_boxes["conv11_2"] * n_classes, kernel_size=3, padding=1
            )

            # Initialize convolutions' parameters
            self.init_conv2d()
        elif backbone_net == "MobileNetV1":
            self.loc_conv4_3 = nn.Conv2d(
                512, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv7 = nn.Conv2d(
                1024, n_boxes["conv7"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv8_2 = nn.Conv2d(
                512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv9_2 = nn.Conv2d(
                256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv10_2 = nn.Conv2d(
                256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1
            )
            self.loc_conv11_2 = nn.Conv2d(
                256, n_boxes["conv11_2"] * 4, kernel_size=3, padding=1
            )

            # Class prediction convolutions (predict classes in localization boxes)
            self.cl_conv4_3 = nn.Conv2d(
                512, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv7 = nn.Conv2d(
                1024, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv8_2 = nn.Conv2d(
                512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv9_2 = nn.Conv2d(
                256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv10_2 = nn.Conv2d(
                256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1
            )
            self.cl_conv11_2 = nn.Conv2d(
                256, n_boxes["conv11_2"] * n_classes, kernel_size=3, padding=1
            )

            # Initialize convolutions' parameters
            self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)

    def forward(
        self,
        conv4_3_feats,
        conv7_feats,
        conv8_2_feats,
        conv9_2_feats,
        conv10_2_feats,
        conv11_2_feats,
    ):
        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        l_conv4_3 = l_conv4_3.view(
            batch_size, -1, 4
        )  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(
            batch_size, -1, 4
        )  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(
            batch_size, -1, self.n_classes
        )  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(
            batch_size, -1, self.n_classes
        )  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(
            batch_size, -1, self.n_classes
        )  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(
            batch_size, -1, self.n_classes
        )  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(
            batch_size, -1, self.n_classes
        )  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(
            batch_size, -1, self.n_classes
        )  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat(
            [l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1
        )  # (N, 8732, 4)
        classes_scores = torch.cat(
            [c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1
        )  # (N, 8732, n_classes)

        return locs, classes_scores


# auxiliary_conv = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]


class AuxillaryConvolutions(nn.Module):
    def __init__(self, backbone_net):
        super(AuxillaryConvolutions, self).__init__()

        if backbone_net == "MobileNetV2":
            self.extras = ModuleList(
                [
                    Sequential(
                        Conv2d(in_channels=1280, out_channels=256, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                ]
            )

            self.init_conv2d()

        elif backbone_net == "MobileNetV1":
            self.extras = ModuleList(
                [
                    Sequential(
                        Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                    Sequential(
                        Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        ReLU(),
                        Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        ReLU(),
                    ),
                ]
            )

            self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            for layer in c:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(c.bias, 0.0)

    def forward(self, inp_features_10x10):
        features = []
        x = inp_features_10x10
        for layer in self.extras:
            x = layer(x)
            features.append(x)

        features_5x5 = features[0]
        features_3x3 = features[1]
        features_2x2 = features[2]
        features_1x1 = features[3]
        return features_5x5, features_3x3, features_2x2, features_1x1


class SSD(nn.Module):
    def __init__(self, num_classes, backbone_network):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.priors = torch.FloatTensor(priors).to(f'npu:{NPU_CALCULATE_DEVICE}')

        # self.base_net = MobileNetV1().model
        self.backbone_net = backbone_network
        if self.backbone_net == "MobileNetV1":
            self.base_net = MobileNetV1().model
        elif self.backbone_net == "MobileNetV2":
            self.base_net = MobileNetV2_pretrained("mobilenet_v2.pth.tar").model
        else:
            raise ("SSD cannot be created with the provided base network")
        # self.base_net = MobileNetV2()

        self.aux_network = AuxillaryConvolutions(self.backbone_net)
        self.prediction_network = PredictionConvolutions(num_classes, self.backbone_net)

    def forward(self, image):

        x = image
        if self.backbone_net == "MobileNetV1":
            source_layer_indexes = [
                12,
                14,
            ]
            start_layer_index = 0
            flag = 0
            x = x.to(f'npu:{NPU_CALCULATE_DEVICE}')
            for end_layer_index in source_layer_indexes:
                for layer in self.base_net[start_layer_index:end_layer_index]:
                    x = layer(x)
                layer_output = x
                start_layer_index = end_layer_index
                if flag == 0:
                    features_19x19 = layer_output
                elif flag == 1:
                    features_10x10 = layer_output
                flag += 1
            for layer in self.base_net[end_layer_index:]:
                x = layer(x)

        elif self.backbone_net == "MobileNetV2":
            for index, feat in enumerate(self.base_net.features):
                x = feat(x)
                if index == 13:
                    features_19x19 = x
                if index == 18:
                    features_10x10 = x

        layer_output = x
        features_5x5, features_3x3, features_2x2, features_1x1 = self.aux_network(
            layer_output
        )

        features = []
        features.append(features_19x19)
        features.append(features_10x10)
        features.append(features_5x5)
        features.append(features_3x3)
        features.append(features_2x2)
        features.append(features_1x1)

        locs, class_scores = self.prediction_network.forward(
            features_19x19,
            features_10x10,
            features_5x5,
            features_3x3,
            features_2x2,
            features_1x1,
        )

        return locs, class_scores


""" 
import numpy as np
import torch
img = np.random.rand(1, 3, 300, 300)
img = torch.Tensor(img)

model = SSD(20)
loc, classes = model.forward(img)
print (loc.shape, classes.shape)
"""
