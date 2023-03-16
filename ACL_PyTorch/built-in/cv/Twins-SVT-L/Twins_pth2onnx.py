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

import argparse
import torch
from timm.models import create_model
# 完成模型的注册
import gvt


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output', default='', help='onnx name')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cpu')
    # 创建模型
    model = create_model('alt_gvt_large', pretrained=False, drop_rate=args.drop,
                         drop_path_rate=args.drop_path, drop_block_rate=None)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)
    dummy_input = torch.randn(1, 3, 224, 224)

    input_names = ["input"]
    output_names = ["modelOutput"]
    dynamic_axes = {'input': {0: 'batch_size'}, 'modelOutput': {0: 'batch_size'}}
    # 开始转换
    torch.onnx.export(model, dummy_input, args.output, input_names=input_names, dynamic_axes=dynamic_axes, 
                      output_names=output_names, opset_version=11, verbose=False)


if __name__ == '__main__':
    args_config = get_args_parser()
    main(args_config)