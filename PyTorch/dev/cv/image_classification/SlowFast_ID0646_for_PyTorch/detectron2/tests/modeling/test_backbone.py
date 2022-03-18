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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

import detectron2.export.torchscript  # apply patch # noqa
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.utils.env import TORCH_VERSION


class TestBackBone(unittest.TestCase):
    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_resnet_scriptability(self):
        cfg = get_cfg()
        resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))

        scripted_resnet = torch.jit.script(resnet)

        inp = torch.rand(2, 3, 100, 100)
        out1 = resnet(inp)["res4"]
        out2 = scripted_resnet(inp)["res4"]
        self.assertTrue(torch.allclose(out1, out2))

    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_fpn_scriptability(self):
        cfg = model_zoo.get_config("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
        bb = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        bb_s = torch.jit.script(bb)

        inp = torch.rand(2, 3, 128, 128)
        out1 = bb(inp)["p5"]
        out2 = bb_s(inp)["p5"]
        self.assertTrue(torch.allclose(out1, out2))
