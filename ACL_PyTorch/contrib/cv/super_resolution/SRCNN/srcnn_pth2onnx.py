#!/usr/bin/python3.7
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

import sys
sys.path.append(r'./SRCNN-pytorch')
from models import SRCNN
import torch
import torch.onnx
import argparse

parser = argparse.ArgumentParser(description='PyTorch Export ONNX')
parser.add_argument('--pth', default='', type=str, metavar='PATH',
                    help='path of pth file (default: none)')
parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
args = parser.parse_args()


def pth2onnx(input_file, output_file):

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(input_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    model.to('cpu')

    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(model, dummy_input, output_file)


if __name__ == '__main__':
    pth2onnx(args.pth, args.onnx)
