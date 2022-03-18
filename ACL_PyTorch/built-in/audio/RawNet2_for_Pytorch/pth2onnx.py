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
import os
import argparse
import yaml
import torch
import onnx
import torch.onnx
from model import RawNet


def convert(pth_model, onnx_model, batch_size):
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml)
    model = RawNet(parser1['model'], 'cpu')
    checkpoint = torch.load(pth_model, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    dummy_input = torch.randn(int(batch_size), 64600)
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, onnx_model,
                      input_names=input_names, output_names=output_names,
                      opset_version=11,
                      dynamic_axes=None,
                      export_params=True,
                      verbose=True,
                      do_constant_folding=True)


def get_parser():
    parser = argparse.ArgumentParser(description='RawNet2')
    parser.add_argument('--pth_model', default=None, type=str,
                        help='Path to pytorch model')
    parser.add_argument('--onnx_model', default=None, type=str,
                        help='Path to onnx model')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Data batch size')
    return parser


if __name__ == "__main__":
    '''
    Example:
        python3.7 pth2onnx.py \
            --pth_model=pre_trained_DF_RawNet2.pth \
            --onnx_model=rawnet2_ori.onnx \
            --batch_size=1
    '''
    parser = get_parser()
    args = parser.parse_args()
    convert(args.pth_model, args.onnx_model, args.batch_size)
    print('pytorch to onnx successfully!')
