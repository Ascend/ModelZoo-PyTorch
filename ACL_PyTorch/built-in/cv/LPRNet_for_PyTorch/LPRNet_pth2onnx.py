# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys

import torch
import torch.onnx as onnx

sys.path.append('LPRNet_Pytorch')

from LPRNet_Pytorch.model import build_lprnet
from LPRNet_Pytorch.data import CHARS


def export_onnx(pth_path, output_path):
    """export onnx model from pytorch model"""
    lprnet = build_lprnet(class_num=len(CHARS))
    device = torch.device('cpu')
    lprnet.to(device)

    checkpoints = torch.load(pth_path, map_location=device)
    lprnet.load_state_dict(checkpoints)

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {"input": {0: '-1'}, "output": {0: '-1'}}

    dummy_input = torch.randn(1, 3, 24, 94, device='cpu')
    onnx.export(lprnet, dummy_input, output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', dest='pth', default='./LPRNet_Pytorch/weights/Final_LPRNet_model.pth',
                        help='pytorch pretrain pth file path')
    parser.add_argument('--output', dest='output', default='./LPRNet.onnx',
                        help='output onnx model file path')
    args = parser.parse_args()
    export_onnx(args.pth, args.output)
    print('Done!')
