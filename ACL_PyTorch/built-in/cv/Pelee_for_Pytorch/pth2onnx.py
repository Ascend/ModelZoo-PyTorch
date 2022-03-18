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
import cv2
import numpy as np
import torch

from configs.CC import Config
import argparse

from peleenet import build_net
from utils.core import *

parser = argparse.ArgumentParser(description='Pelee Testing')
parser.add_argument('-c', '--config', default='configs/Pelee_VOC.py')
parser.add_argument('-m', '--trained_model', default=None,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-o', '--output', default=None,
                    type=str, help='ONNX model file')
args = parser.parse_args()

global cfg
cfg = Config.fromfile(args.config)

model = build_net('test', cfg.model.input_size, cfg.model)
init_net(model, cfg, args.trained_model)

model.eval()
input_names = ["image"]
output_names = ["output1", "output2"]
dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'}}
dummy_input = torch.randn(1, 3, 304, 304)
torch.onnx.export(model, dummy_input, args.output, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=False, opset_version=13)






