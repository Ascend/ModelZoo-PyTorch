# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.onnx

from collections import OrderedDict
import _init_paths
import models
from config import cfg
from config import update_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--weights',
                        help='weights path',
                        default='model_best.pth.tar',
                        type=str)
    # parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')

    args = parser.parse_args()

    return args


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def export_onnx(cfg, weights):

    model = models.pose_higher_hrnet.get_pose_net(cfg,is_train=False)
    model.eval()
    # model = network_to_half(model)

    checkpoint = torch.load(weights, map_location='cpu')
    # checkpoint = torch.load(path)
    onnx_path = weights.with_suffix('.onnx')
    try:
        model.load_state_dict(checkpoint)
    except:
        checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
        model.load_state_dict(checkpoint['state_dict'])

    # print(model)
    # summary(model, (3, 512, 512))
    input_names = ["input"]
    output_names = ["output1","output2"]
    dummy_input = torch.zeros(1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names,
                      output_names=output_names,
                      opset_version=11)

    print('onnx export done .')

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    export_onnx(cfg,args.weights)