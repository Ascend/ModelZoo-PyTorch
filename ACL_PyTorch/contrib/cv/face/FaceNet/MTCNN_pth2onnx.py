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
sys.path.append('./models')
from mtcnn import PNet_truncated, RNet_truncated, ONet_truncated


def MTCNN_pth2onnx(opt):
    if opt.model == 'PNet':
        model = PNet_truncated()
    elif opt.model == 'RNet':
        model = RNet_truncated()
    elif opt.model == 'ONet':
        model = ONet_truncated()
    else:
        print("Error network")
        return -1
    model = model.eval()
    input_names = ['image']
    if opt.model == 'PNet':
        output_names = ["probs", "reg"]
        dynamic_axes = {'image': {0: '-1', 2: '-1', 3: '-1'}, 'probs': {0: '-1', 2: '-1', 3: '-1'},
                        'reg': {0: '-1', 2: '-1', 3: '-1'}}
        dummy_input = torch.randn(1, 3, 1229, 1000)
    elif opt.model == 'RNet':
        output_names = ['regs', 'cls']
        dynamic_axes = {'image': {0: '-1'}, 'regs': {0: '-1'}, 'cls': {0: '-1'}}
        dummy_input = torch.randn(20, 3, 24, 24)
    else:
        output_names = ['landmark', 'regs', 'cls']
        dynamic_axes = {'image': {0: '-1'}, 'landmark': {0: '-1'}, 'regs': {0: '-1'}, 'cls': {0: '-1'}}
        dummy_input = torch.randn(20, 3, 48, 48)

    torch.onnx.export(model, dummy_input, opt.output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, verbose=True, opset_version=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='[PNet/RNet/ONet]')
    parser.add_argument('--output_file', type=str, default='.', help='output path')
    arg = parser.parse_args()
    MTCNN_pth2onnx(arg)
