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
import torch
import argparse

sys.path.append('RCAN/RCAN_TestCode/code')
from model.rcan import RCAN

parser = argparse.ArgumentParser(description='PyTorch Export ONNX')
parser.add_argument('--pth', default='', type=str, metavar='PATH',
                    help='path of pth file (default: none)')
parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
parser.add_argument('--shape', default='1,3,256,256', help='dummy input shape')
args = parser.parse_args()

class Margs:
    def __init__(self, n_resgroups, n_resblocks, n_feats, reduction, scale, data_train, rgb_range, res_scale, n_colors):
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.scale = scale
        self.data_train = data_train
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.n_colors = n_colors


if __name__ == "__main__":

    input_names = ["image"]
    output_names = ["HR_image"]
    dynamic_axes = {'image': {0: '-1'}, 'HR_image': {0: '-1'}}

    dummy_input_str = "torch.randn({}, device='cpu')".format(args.shape)

    # dummy_input = torch.randn(1, 3, 256, 256, device='cpu')
    dummy_input = eval(dummy_input_str)

    model_args = Margs(10, 20, 64, 16, [2,], 'DIV2K', 255, 1, 3)

    model = RCAN(model_args)

    model.load_state_dict(torch.load(args.pth, 'cpu'),strict=True)
    model.eval()
    torch.onnx.export(model, dummy_input, args.onnx, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=True)




    