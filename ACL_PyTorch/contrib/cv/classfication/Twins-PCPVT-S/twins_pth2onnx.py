# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import torch.onnx
import onnx
import torch
import gvt
from timm.models import create_model

def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='pcpvt_small_v0', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--source', default='./pcpvt_small.pth', help='resume from checkpoint')
    parser.add_argument('--target', default='./twins_dynamic.onnx', help='onnx save path')

    return parser

def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.source, map_location='cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.to(device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    torch.onnx.export(model, dummy_input, args.target, verbose=True, 
                      input_names=input_names, dynamic_axes=dynamic_axes, 
                      output_names=output_names)

    # 增加维度信息
    model_file = args.target
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Twins training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
