# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
from collections import OrderedDict
import torch
import torch.onnx
sys.path.append(r"./HRNet-Image-Classification")
sys.path.append(r"./HRNet-Image-Classification/lib")
from lib.models import cls_hrnet
from lib.config import config
from lib.config import update_config
def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
                        
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    parser.add_argument('--input',
                        help='input pytorch model',
                        required=True,
                        type=str)
    
    parser.add_argument('--output',
                        help='output onnx model',
                        required=True,
                        type=str)

    args = parser.parse_args()
    update_config(config, args)

    return args

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx():
    args = parse_args()
    print(config.MODEL)
    modelpth = args.input
    checkpoint = torch.load(modelpth, map_location='cpu')
    model = cls_hrnet.get_cls_net(config)
    output_file = args.output
    print("output:",output_file)
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, 
                    input_names = input_names, dynamic_axes = dynamic_axes, 
                    output_names = output_names, verbose=True, opset_version=11)

if __name__ == "__main__":
    pth2onnx()
