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

import os
import sys
import torch
import torch.onnx

from collections import OrderedDict
sys.path.append('./TransPose')
import transpose_r
from lib.config import cfg
from lib.config import update_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='TransPose/experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--weights',
                        help='weights path',
                        default='model_best.pth.tar',
                        type=str)
    parser.add_argument('--bs', type=int, default=1)
    opt = parser.parse_args()

    return opt


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def export_onnx(config, weights, bs):

    model = transpose_r.get_pose_net(config, is_train=False)
    model.eval()

    checkpoint = torch.load(weights, map_location='cpu')
    onnx_path = os.path.splitext(weights)[0] + ".onnx"
    try:
        model.load_state_dict(checkpoint)
    except:
        checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
        model.load_state_dict(checkpoint['state_dict'])

    input_names = ["input"]
    output_names = ["output"]
    dummy_input = torch.zeros(bs, 3, 256, 192)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names,
                      dynamic_axes={'input': {0: 'bs'}},
                      output_names=output_names,
                      opset_version=11)
    print('onnx export done .')


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    export_onnx(cfg, args.weights, args.bs)
