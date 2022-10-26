# Copyright 2022 Huawei Technologies Co., Ltd
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
sys.path.append("./DEKR")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing
import torch.onnx
from collections import OrderedDict

from tools import _init_paths
import models

from lib.config import cfg
from lib.config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./DEKR/experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml',
                        type=str)

    parser.add_argument('--output',
                        default='models/dekr_bs1.onnx',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    device = torch.device('cpu')

    checkpoint = torch.load(cfg.TEST.MODEL_FILE, map_location=device)
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint, strict=True)

    model.eval()
    output_file = args.output
    pth2onnx(model, output_file)



def pth2onnx(model, output_file):
    model.eval()
    input_names = ["image"]
    output_names = ["heatmap", "offset"]
    dynamic_axes = {'image': {2: '-1', 3:'-1'}, 'heatmap': {0: '-1'},'offset':{0:'-1'}}
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, output_file, input_names= input_names, dynamic_axes= None, output_names= output_names, opset_version=11, verbose=False, enable_onnx_checker=False)


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    main()
