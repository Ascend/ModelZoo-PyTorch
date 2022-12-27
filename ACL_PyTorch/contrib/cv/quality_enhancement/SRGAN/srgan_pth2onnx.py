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

import argparse

import torch

from model import Generator

def pth2onnx(args):
    netG = Generator(scale_factor=args.upscale_factor)
    netG.load_state_dict(torch.load(args.src_path, map_location='cpu'))
    netG.eval()
    input_names = ["lrImage"]
    output_names = ["hrImage"]
    
    dummy_input = torch.randn(16, 3, 400, 400)
    dynamic_axes = {'lrImage': {0: '-1'}}
    export_name = args.result_path
    torch.onnx.export(netG, dummy_input, export_name,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fix onnx script')
    parser.add_argument('--src_path', default='./netG_best.pth', type=str, help='weight path')
    parser.add_argument('--result_path', default='./srgan.onnx', type=str, help='onnx path')
    parser.add_argument('--upscale_factor', default=2, type=int, help='upscale_factor')
    args_parser = parser.parse_args()
    
    pth2onnx(args_parser)
