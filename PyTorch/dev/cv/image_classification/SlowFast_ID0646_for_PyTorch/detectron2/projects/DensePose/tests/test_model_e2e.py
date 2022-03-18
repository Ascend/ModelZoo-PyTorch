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

from detectron2.structures import BitMasks, Boxes, Instances

from .common import get_model


# TODO(plabatut): Modularize detectron2 tests and re-use
def make_model_inputs(image, instances=None):
    if instances is None:
        return {"image": image}

    return {"image": image, "instances": instances}


def make_empty_instances(h, w):
    instances = Instances((h, w))
    instances.gt_boxes = Boxes(torch.rand(0, 4))
    instances.gt_classes = torch.tensor([]).to(dtype=torch.int64)
    instances.gt_masks = BitMasks(torch.rand(0, h, w))
    return instances


class ModelE2ETest(unittest.TestCase):
    CONFIG_PATH = ""

    def setUp(self):
        self.model = get_model(self.CONFIG_PATH)

    def _test_eval(self, sizes):
        inputs = [make_model_inputs(torch.rand(3, size[0], size[1])) for size in sizes]
        self.model.eval()
        self.model(inputs)


class DensePoseRCNNE2ETest(ModelE2ETest):
    CONFIG_PATH = "densepose_rcnn_R_101_FPN_s1x.yaml"

    def test_empty_data(self):
        self._test_eval([(200, 250), (200, 249)])
