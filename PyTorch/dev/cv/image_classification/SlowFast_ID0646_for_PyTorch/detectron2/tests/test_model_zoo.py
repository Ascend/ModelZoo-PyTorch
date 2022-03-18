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

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling import FPN, GeneralizedRCNN

logger = logging.getLogger(__name__)


class TestModelZoo(unittest.TestCase):
    def test_get_returns_model(self):
        model = model_zoo.get("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", trained=False)
        self.assertIsInstance(model, GeneralizedRCNN)
        self.assertIsInstance(model.backbone, FPN)

    def test_get_invalid_model(self):
        self.assertRaises(RuntimeError, model_zoo.get, "Invalid/config.yaml")

    def test_get_url(self):
        url = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
        self.assertEqual(
            url,
            "https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn/138602908/model_final_01ca85.pkl",  # noqa
        )
        url2 = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.py")
        self.assertEqual(url, url2)

    def _build_lazy_model(self, name):
        cfg = model_zoo.get_config("common/models/" + name)
        instantiate(cfg.model)

    def test_mask_rcnn_fpn(self):
        self._build_lazy_model("mask_rcnn_fpn.py")

    def test_mask_rcnn_c4(self):
        self._build_lazy_model("mask_rcnn_c4.py")

    def test_panoptic_fpn(self):
        self._build_lazy_model("panoptic_fpn.py")

    def test_schedule(self):
        cfg = model_zoo.get_config("common/coco_schedule.py")
        for _, v in cfg.items():
            instantiate(v)


if __name__ == "__main__":
    unittest.main()
