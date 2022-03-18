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

import os
import sys
import torch
from torch import nn

from model import Generator

def pth2onnx(input_file, upscale_factor):
    device = torch.device('cpu')
    # 创建模型
    netG = Generator(scale_factor=upscale_factor)
    # 加载参数
    netG.load_state_dict(torch.load(input_file, map_location='cpu'))
    netG.eval()
    input_names = ["lrImage"]
    output_names = ["hrImage"]
    
    model_name ='SRGAN'
    dummy_input = torch.randn(1, 3, 400, 400)
    export_name = 'srgan.onnx'
    torch.onnx.export(netG, dummy_input, export_name,
                      input_names=input_names, output_names=output_names,
                      opset_version=11)

if __name__ == '__main__':
    pth2onnx(sys.argv[1], 2)
