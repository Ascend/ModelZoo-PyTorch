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

from collections import OrderedDict
import sys

sys.path.append('./xcit')
import argparse
import torch
from timm.models import create_model
from onnx_test import onnx_align
import xcit


def get_args_parser():
    parser = argparse.ArgumentParser('XCiT pth2onnx script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.add_argument('--model', default='xcit_small_12_p16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Dataset parameters
    parser.add_argument('--data-path', type=str, help='dataset path')
    parser.add_argument('--on_cpu', default=True, help='put model on cpu')
    parser.add_argument("--pretrained", default=None, type=str, help='Path to pre-trained checkpoint')
    # onnx parameters
    parser.add_argument('--fp16', action='store_true', help='enable for onnx fp16 transfer')
    return parser


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def model_transfer(args):
    batch_size = args.batch_size
    onnx_model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        on_cpu=args.on_cpu,
        fp16=args.fp16
    )
    ckpt = torch.load(args.pretrained, map_location='cpu')
    ckpt['model'] = proc_nodes_module(ckpt, 'model')
    onnx_model.load_state_dict(ckpt['model'])
    onnx_model.cpu()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    if args.fp16:
        onnx_path = 'onnx_models/xcit_b' + str(batch_size) + '_fp16.onnx'
    else:
        onnx_path = 'onnx_models/xcit_b' + str(batch_size) + '.onnx'
    torch.onnx.export(onnx_model, dummy_input, onnx_path, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)
    print("model saved successful at ", onnx_path)
    onnx_align(onnx_model, onnx_path, batch_size, args.fp16)  # config.ONNX_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('XCiT pth2om script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.nb_classes = 1000
    model_transfer(args)
