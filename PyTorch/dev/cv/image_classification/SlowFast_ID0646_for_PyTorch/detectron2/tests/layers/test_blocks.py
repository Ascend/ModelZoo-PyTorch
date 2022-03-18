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
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
import torch
from torch import nn

from detectron2.layers import ASPP, DepthwiseSeparableConv2d, FrozenBatchNorm2d
from detectron2.modeling.backbone.resnet import BasicStem, ResNet


"""
Test for misc layers.
"""


class TestBlocks(unittest.TestCase):
    def test_separable_conv(self):
        DepthwiseSeparableConv2d(3, 10, norm1="BN", activation1=nn.PReLU())

    def test_aspp(self):
        m = ASPP(3, 10, [2, 3, 4], norm="", activation=nn.PReLU())
        self.assertIsNot(m.convs[0].activation.weight, m.convs[1].activation.weight)
        self.assertIsNot(m.convs[0].activation.weight, m.project.activation.weight)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_frozen_batchnorm_fp16(self):
        from torch.cuda.amp import autocast

        C = 10
        input = torch.rand(1, C, 10, 10).cuda()
        m = FrozenBatchNorm2d(C).cuda()
        with autocast():
            output = m(input.half())
        self.assertEqual(output.dtype, torch.float16)

        # requires_grad triggers a different codepath
        input.requires_grad_()
        with autocast():
            output = m(input.half())
        self.assertEqual(output.dtype, torch.float16)

    def test_resnet_unused_stages(self):
        resnet = ResNet(BasicStem(), ResNet.make_default_stages(18), out_features=["res2"])
        self.assertTrue(hasattr(resnet, "res2"))
        self.assertFalse(hasattr(resnet, "res3"))
        self.assertFalse(hasattr(resnet, "res5"))

        resnet = ResNet(BasicStem(), ResNet.make_default_stages(18), out_features=["res2", "res5"])
        self.assertTrue(hasattr(resnet, "res2"))
        self.assertTrue(hasattr(resnet, "res4"))
        self.assertTrue(hasattr(resnet, "res5"))
