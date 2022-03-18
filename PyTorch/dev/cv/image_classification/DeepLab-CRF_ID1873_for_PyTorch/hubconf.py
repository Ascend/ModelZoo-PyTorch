#!/usr/bin/env python
# coding: utf-8
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
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   20 December 2018

from __future__ import print_function

from torch.hub import load_state_dict_from_url
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

model_url_root = "https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/"
model_dict = {
    "cocostuff10k": ("deeplabv2_resnet101_msc-cocostuff10k-20000.pth", 182),
    "cocostuff164k": ("deeplabv2_resnet101_msc-cocostuff164k-100000.pth", 182),
    "voc12": ("deeplabv2_resnet101_msc-vocaug-20000.pth", 21),
}


def deeplabv2_resnet101(pretrained=None, n_classes=182, scales=None):

    from libs.models.deeplabv2 import DeepLabV2
    from libs.models.msc import MSC

    # Model parameters
    n_blocks = [3, 4, 23, 3]
    atrous_rates = [6, 12, 18, 24]
    if scales is None:
        scales = [0.5, 0.75]

    base = DeepLabV2(n_classes=n_classes, n_blocks=n_blocks, atrous_rates=atrous_rates)
    model = MSC(base=base, scales=scales)

    # Load pretrained models
    if isinstance(pretrained, str):

        assert pretrained in model_dict, list(model_dict.keys())
        expected = model_dict[pretrained][1]
        error_message = "Expected: n_classes={}".format(expected)
        assert n_classes == expected, error_message

        model_url = model_url_root + model_dict[pretrained][0]
        state_dict = load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict)

    return model

