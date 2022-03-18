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
import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, RotatedAnchorGenerator

logger = logging.getLogger(__name__)


class TestAnchorGenerator(unittest.TestCase):
    def test_default_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 1, 4]]

        anchor_generator = DefaultAnchorGenerator(cfg, [ShapeSpec(stride=4)])

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        anchors = anchor_generator([features["stage3"]])
        expected_anchor_tensor = torch.tensor(
            [
                [-32.0, -8.0, 32.0, 8.0],
                [-16.0, -16.0, 16.0, 16.0],
                [-8.0, -32.0, 8.0, 32.0],
                [-64.0, -16.0, 64.0, 16.0],
                [-32.0, -32.0, 32.0, 32.0],
                [-16.0, -64.0, 16.0, 64.0],
                [-28.0, -8.0, 36.0, 8.0],  # -28.0 == -32.0 + STRIDE (4)
                [-12.0, -16.0, 20.0, 16.0],
                [-4.0, -32.0, 12.0, 32.0],
                [-60.0, -16.0, 68.0, 16.0],
                [-28.0, -32.0, 36.0, 32.0],
                [-12.0, -64.0, 20.0, 64.0],
            ]
        )

        self.assertTrue(torch.allclose(anchors[0].tensor, expected_anchor_tensor))

    def test_default_anchor_generator_centered(self):
        # test explicit args
        anchor_generator = DefaultAnchorGenerator(
            sizes=[32, 64], aspect_ratios=[0.25, 1, 4], strides=[4]
        )

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        expected_anchor_tensor = torch.tensor(
            [
                [-30.0, -6.0, 34.0, 10.0],
                [-14.0, -14.0, 18.0, 18.0],
                [-6.0, -30.0, 10.0, 34.0],
                [-62.0, -14.0, 66.0, 18.0],
                [-30.0, -30.0, 34.0, 34.0],
                [-14.0, -62.0, 18.0, 66.0],
                [-26.0, -6.0, 38.0, 10.0],
                [-10.0, -14.0, 22.0, 18.0],
                [-2.0, -30.0, 14.0, 34.0],
                [-58.0, -14.0, 70.0, 18.0],
                [-26.0, -30.0, 38.0, 34.0],
                [-10.0, -62.0, 22.0, 66.0],
            ]
        )

        anchors = anchor_generator([features["stage3"]])
        self.assertTrue(torch.allclose(anchors[0].tensor, expected_anchor_tensor))

        anchors = torch.jit.script(anchor_generator)([features["stage3"]])
        self.assertTrue(torch.allclose(anchors[0].tensor, expected_anchor_tensor))

    def test_rrpn_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 1, 4]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [0, 45]  # test single list[float]
        anchor_generator = RotatedAnchorGenerator(cfg, [ShapeSpec(stride=4)])

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        anchors = anchor_generator([features["stage3"]])
        expected_anchor_tensor = torch.tensor(
            [
                [0.0, 0.0, 64.0, 16.0, 0.0],
                [0.0, 0.0, 64.0, 16.0, 45.0],
                [0.0, 0.0, 32.0, 32.0, 0.0],
                [0.0, 0.0, 32.0, 32.0, 45.0],
                [0.0, 0.0, 16.0, 64.0, 0.0],
                [0.0, 0.0, 16.0, 64.0, 45.0],
                [0.0, 0.0, 128.0, 32.0, 0.0],
                [0.0, 0.0, 128.0, 32.0, 45.0],
                [0.0, 0.0, 64.0, 64.0, 0.0],
                [0.0, 0.0, 64.0, 64.0, 45.0],
                [0.0, 0.0, 32.0, 128.0, 0.0],
                [0.0, 0.0, 32.0, 128.0, 45.0],
                [4.0, 0.0, 64.0, 16.0, 0.0],  # 4.0 == 0.0 + STRIDE (4)
                [4.0, 0.0, 64.0, 16.0, 45.0],
                [4.0, 0.0, 32.0, 32.0, 0.0],
                [4.0, 0.0, 32.0, 32.0, 45.0],
                [4.0, 0.0, 16.0, 64.0, 0.0],
                [4.0, 0.0, 16.0, 64.0, 45.0],
                [4.0, 0.0, 128.0, 32.0, 0.0],
                [4.0, 0.0, 128.0, 32.0, 45.0],
                [4.0, 0.0, 64.0, 64.0, 0.0],
                [4.0, 0.0, 64.0, 64.0, 45.0],
                [4.0, 0.0, 32.0, 128.0, 0.0],
                [4.0, 0.0, 32.0, 128.0, 45.0],
            ]
        )

        self.assertTrue(torch.allclose(anchors[0].tensor, expected_anchor_tensor))


if __name__ == "__main__":
    unittest.main()
