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
import argparse

import torch
import torch.onnx
from collections import OrderedDict
import sys
sys.path.append('./HigherHRNet-Human-Pose-Estimation')
from lib.config import update_config
from lib.config import cfg
from lib.models import pose_higher_hrnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--weights',
                        help='weights path',
                        default='model_best.pth.tar',
                        type=str)
    parser.add_argument('--onnx_path',
                        help='onnx path',
                        default='pose_higher_hrnet_w32_512_bs1_dynamic.onnx',
                        type=str)
    parser.add_argument('--bs', type=int, default=1)

    return parser.parse_args()


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def export_onnx(config, weights, onnx_path, bs):
    model = pose_higher_hrnet.get_pose_net(config, is_train=False)
    model.eval()
    checkpoint = torch.load(weights, map_location='cpu')

    try:
        model.load_state_dict(checkpoint)
    except:
        checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
        model.load_state_dict(checkpoint['state_dict'])

    input_names = ["input"]
    output_names = ["output1", "output2"]
    dummy_input = torch.zeros(
        bs, 3, config.DATASET.INPUT_SIZE, config.DATASET.INPUT_SIZE)
    dynamic_axes = {"input": {0: "-1", 2: "-1", 3: "-1"},
                    "output1": {0: "-1"},
                    "output2": {0: "-1"}}

    torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11)

    print('onnx export done .')


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    export_onnx(cfg, args.weights, args.onnx_path, args.bs)
