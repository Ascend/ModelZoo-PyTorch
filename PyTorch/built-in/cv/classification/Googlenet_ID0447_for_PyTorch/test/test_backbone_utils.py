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
#import unittest


import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ResnetFPNBackboneTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32

    def test_resnet18_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet18_fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)
        y = resnet18_fpn(x)
        assert list(y.keys()) == [0, 1, 2, 3, 'pool']

    def test_resnet50_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet50_fpn = resnet_fpn_backbone(backbone_name='resnet50', pretrained=False)
        y = resnet50_fpn(x)
        assert list(y.keys()) == [0, 1, 2, 3, 'pool']
