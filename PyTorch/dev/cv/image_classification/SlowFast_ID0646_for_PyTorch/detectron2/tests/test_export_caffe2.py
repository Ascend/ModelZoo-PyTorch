# Copyright (c) Facebook, Inc. and its affiliates.
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

import copy
import os
import tempfile
import unittest
import torch

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.testing import get_sample_coco_image


@unittest.skipIf(os.environ.get("CI"), "Require COCO data and model zoo.")
class TestCaffe2Export(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _test_model(self, config_path, device="cpu"):
        # requires extra dependencies
        from detectron2.export import Caffe2Model, add_export_config, Caffe2Tracer

        cfg = model_zoo.get_config(config_path)
        add_export_config(cfg)
        cfg.MODEL.DEVICE = device
        model = model_zoo.get(config_path, trained=True, device=device)

        inputs = [{"image": get_sample_coco_image()}]
        tracer = Caffe2Tracer(cfg, model, copy.deepcopy(inputs))

        c2_model = tracer.export_caffe2()

        with tempfile.TemporaryDirectory(prefix="detectron2_unittest") as d:
            c2_model.save_protobuf(d)
            c2_model.save_graph(os.path.join(d, "test.svg"), inputs=copy.deepcopy(inputs))

            c2_model = Caffe2Model.load_protobuf(d)
            c2_model(inputs)[0]["instances"]

            ts_model = tracer.export_torchscript()
            ts_model.save(os.path.join(d, "model.ts"))

    def testMaskRCNN(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNNGPU(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device="cuda")

    def testRetinaNet(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    def testPanopticFPN(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
