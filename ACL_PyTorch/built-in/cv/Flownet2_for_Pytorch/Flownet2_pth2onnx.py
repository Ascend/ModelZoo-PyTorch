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
import argparse
import torch
sys.path.append('./flownet2-pytorch')
import models
from utils import tools


def parser_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--input_path', type=str, default='./FlowNet2_checkpoint.pth.tar')
    parser.add_argument('--out_path', type=str, default='./models/flownet2_bs1.onnx')
    parser.add_argument('--model', type=str, default='FlowNet2')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


def export_onnx():
    model_class = tools.module_to_dict(models)[args.model]
    model = model_class(args)
    checkpoint = torch.load(args.input_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_names = ['x1', 'x2']
    output_names = ['flow']
    dummy_input_x1 = torch.randn(args.batch_size, 3, 448, 1024)
    dummy_input_x2 = torch.randn(args.batch_size, 3, 448, 1024)

    torch.onnx.export(model, (dummy_input_x1, dummy_input_x2), args.out_path,
                      input_names=input_names, output_names=output_names, opset_version=11,
                      verbose=True)


if __name__ == '__main__':
    args = parser_func()
    export_onnx()
