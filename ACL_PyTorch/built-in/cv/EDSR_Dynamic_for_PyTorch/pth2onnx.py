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
import argparse
import torch


sys.path.append('./EDSR-PyTorch/src')
from model.edsr import EDSR


def parser_func():
    parser = argparse.ArgumentParser(description='EDSR onnx export.')
    parser.add_argument('-i', '--input_path', type=str, default='./EDSR_x2.pt',
                        help='input path for pth model')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--n_resblocks', type=int, default=3,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=256,
                        help='numbe of feature maps')
    parser.add_argument('--res_scale', type=float, default=0.1,
                        help='residual scaling')
    parser.add_argument('-o', '--out_path', type=str, default='./models/onnx/EDSR_x2.onnx',
                        help='out path for onnx model')
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    return args


def pth2onnx():
    model = EDSR(args)
    model.load_state_dict(torch.load(args.input_path, map_location='cpu'), strict=False)
    dynamic_axes = {
        'image': {0: 'B', 2: 'H', 3: 'W'},
        'out': {0: '-1', 2: '-1', 3: '-1'}
    }
    input_data = torch.randn([1, 3, 256, 256]).to(torch.float32)
    input_names = ["image"]
    output_names = ["out"]

    torch.onnx.export(
        model,
        input_data,
        args.out_path,
        dynamic_axes=dynamic_axes,
        verbose=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names
    )


if __name__ == '__main__':
    args = parser_func()
    pth2onnx()
