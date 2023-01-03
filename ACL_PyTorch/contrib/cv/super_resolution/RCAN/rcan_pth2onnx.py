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


import sys
import argparse

import torch

sys.path.append('RCAN/RCAN_TestCode/code')
from model.rcan import RCAN


class Margs:
    def __init__(self, n_resgroups, n_resblocks, n_feats, reduction, 
                 scale, data_train, rgb_range, res_scale, n_colors):
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.scale = scale
        self.data_train = data_train
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.n_colors = n_colors


def convert(ckpt_path, onnx_path, input_shape, scale):
    model_args = Margs(10, 20, 64, 16, [scale,], 'DIV2K', 255, 1, 3)
    model = RCAN(model_args)
    model.load_state_dict(torch.load(ckpt_path, 'cpu'), strict=True)
    model.eval()

    dummy_input = torch.randn((1, 3, *input_shape), device='cpu')
    input_names = ["image"]
    output_names = ["HR_image"]
    dynamic_axes = {'image': {0: '-1'}, 'HR_image': {0: '-1'}}
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=input_names, 
                      dynamic_axes=dynamic_axes, 
                      output_names=output_names, 
                      verbose=False,
                      opset_version=13)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch export to ONNX')
    parser.add_argument('--pth', type=str, metavar='PATH',
                        help='path of pth file')
    parser.add_argument('--onnx', type=str, metavar='PATH',
                        help='path to save output onnx model.')
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256], 
                        help='the height and width of model input.')
    parser.add_argument('--scale', type=int, default=2, 
                        help='the magnifying rates of output images.')
    args = parser.parse_args()

    convert(args.pth, args.onnx, args.shape, args.scale)
