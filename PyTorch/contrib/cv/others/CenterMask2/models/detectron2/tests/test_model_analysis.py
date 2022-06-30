# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import torch

import detectron2.model_zoo as model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.analysis import flop_count_operators, parameter_count


def get_model_zoo(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg_file = model_zoo.get_config_file(config_path)
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return build_model(cfg)


class RetinaNetTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    def test_flop(self):
        # RetinaNet supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)
        self.assertTrue(int(res["conv"]), 146)  # 146B flops

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 37915572)
        self.assertTrue(res["backbone"], 31452352)


class FasterRCNNTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    def test_flop(self):
        # Faster R-CNN supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)

        # This only checks flops for backbone & proposal generator
        # Flops for box head is not conv, and depends on #proposals, which is
        # almost 0 for random inputs.
        self.assertTrue(int(res["conv"]), 117)

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 41699936)
        self.assertTrue(res["backbone"], 26799296)
