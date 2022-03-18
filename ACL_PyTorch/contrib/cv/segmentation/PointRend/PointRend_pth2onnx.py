# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import sys
from collections import OrderedDict
import torch
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

def setup(cfg_file):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg

def pth2onnx(cfg_file, pth_file, output_file):
    cfg = setup(cfg_file)
    model = DefaultTrainer.build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(pth_file)
    model.eval()

    input_name = ['images']
    output_name = ['results']
    dynamic_axes = {'images': {0: '-1'}, 'results': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 1024, 2048)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_name, dynamic_axes=dynamic_axes, output_names=output_name, opset_version=11, verbose=True)

if __name__ == '__main__':
    cfg_file = sys.argv[1]
    pth_file = sys.argv[2]
    output_file = sys.argv[3]
    pth2onnx(cfg_file, pth_file, output_file)
