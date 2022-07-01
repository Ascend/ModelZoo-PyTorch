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
"""MobileNetv2 Encoder"""

import torch
import torch.nn as nn

from .layer_factory import InvertedResidual, conv_bn_relu6
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


__all__ = ["mbv2"]


model_paths = {"mbv2_voc": "./data/weights/mbv2_voc_rflw.ckpt"}


class MobileNetV2(nn.Module):
    """MobileNetV2 definition"""

    # expansion rate, output channels, number of repeats, stride
    mobilenet_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, width_mult=1.0, return_layers=[1, 2, 4, 6]):
        super(MobileNetV2, self).__init__()
        self.return_layers = return_layers
        self.max_layer = max(self.return_layers)
        self.out_sizes = [
            self.mobilenet_config[layer_idx][1] for layer_idx in self.return_layers
        ]
        input_channel = int(self.in_planes * width_mult)
        self.layer1 = conv_bn_relu6(3, input_channel, 2)
        for layer_idx, (t, c, n, s) in enumerate(
            self.mobilenet_config[: self.max_layer + 1]
        ):
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(
                        InvertedResidual(input_channel, output_channel, s, t)
                    )
                else:
                    features.append(
                        InvertedResidual(input_channel, output_channel, 1, t)
                    )
                input_channel = output_channel
            setattr(self, "layer{}".format(layer_idx + 2), nn.Sequential(*features))

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        for layer_idx in range(self.max_layer + 1):
            x = getattr(self, "layer{}".format(layer_idx + 2))(x)
            outs.append(x)
        return [outs[layer_idx] for layer_idx in self.return_layers]


def mbv2(pretrained=False, **kwargs):
    """Constructs a MobileNet-v2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.load(model_paths["mbv2_{}".format(str(pretrained))]), strict=False
        )
    return model


def create_encoder(pretrained="voc", ctrl_version="cvpr", **kwargs):
    """Create Encoder"""
    return_layers = [1, 2, 4, 6] if ctrl_version == "cvpr" else [1, 2]
    return mbv2(pretrained=pretrained, return_layers=return_layers, **kwargs)
