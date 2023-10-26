# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
# -*- coding: utf-8 -*-

import os
import sys
import argparse

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.onnx

from collections import OrderedDict
from enet import get_enet


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx(input_file, output_file, batch_size=1):
    model = get_enet(model='enet', dataset='citys', aux=False, norm_layer=nn.BatchNorm2d)
    checkpoint = {}
    checkpoint['state_dict'] = torch.load(input_file, map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    print(model)

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(batch_size, 3, 480, 480)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, 
                        output_names = output_names, opset_version=11, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='~/.torch/models/enet_citys.pth')
    parser.add_argument('--output-file', type=str, default='model/enet_citys_bs1.onnx')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    pth2onnx(args.input_file, args.output_file, batch_size=args.batch_size)
