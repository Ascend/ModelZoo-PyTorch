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
sys.path.append(r'./EDSR-PyTorch/src')
import torch
import torch.onnx
import argparse
import model
import utility

parser = argparse.ArgumentParser(description='PyTorch Export ONNX')
parser.add_argument('--pth', default='', type=str, metavar='PATH',
                    help='path of pth file (default: none)')
parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
parser.add_argument('--size', default=1020, type=int,
                    help='output size')
args = parser.parse_args()

# default args for running model
class Margs:
    def __init__(self):
        self.scale = [2]
        self.pre_train = args.pth
        self.test_only = True
        self.cpu = True
        self.load = ''
        self.save = ''
        self.reset = ''
        self.data_test = ''
        self.model = 'EDSR'
        self.self_ensemble = False
        self.chop = False
        self.precision = 'single'
        self.n_GPUs = 1
        self.save_models = False
        self.n_resblocks = 16
        self.n_feats = 64
        self.rgb_range = 255
        self.n_colors = 3
        self.res_scale = 1
        self.resume = 0

checkpoint = utility.checkpoint(Margs())

def pth2onnx(input_file, output_file,size):

    _model = model.Model(Margs(), checkpoint)

    _model.load_state_dict(torch.load(
        input_file, map_location=torch.device('cpu')), strict=False)
    _model.eval()

    dummy_input = torch.randn(1, 3, size, size)
    torch.onnx.export(_model, dummy_input, output_file,
                      opset_version=9, verbose=False)


if __name__ == '__main__':
    pth2onnx(args.pth, args.onnx, args.size)
