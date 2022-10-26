# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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

import torch
import torchvision
import sys
sys.path.append(r"./Spach")
from models.spach.spach_ms import SpachMS
from collections import OrderedDict
import argparse


def pth2onnx(input_file, output_file):

    cfgs = dict(img_size=224, patch_size=4, hidden_dim=128, token_ratio=0.5, num_heads=4, channel_ratio=3.0)
    cfgs['net_arch'] = [[('pass', 3)], [('pass', 4)], [('pass', 12)], [('pass', 3)]]
    model = SpachMS(**cfgs)
    checkpoint = torch.load(input_file, map_location='cpu')

    # 加载模型参数
    checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    torch.onnx.export(model, dummy_input, output_file, verbose=True, input_names=input_names, dynamic_axes=dynamic_axes,
                      opset_version=11,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--input-path', type=str, default = None)
    parser.add_argument('--output-path', type=str, default = None)
    args = parser.parse_args()
    pth2onnx(args.input_path, args.output_path)


