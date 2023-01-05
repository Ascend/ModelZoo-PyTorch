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
import importlib
import sys

import torch

def initParams(parser):
    parser.set_defaults(
        num_channels=3,
        num_patches=1000,
        train_batch_size=16,
        eval_batch_size=1,
        image_mean=0.5,
        scale=2,
        dataset='div2k'
    )

def pth2onnx(parser):
    args, _ = parser.parse_known_args()
    initParams(parser)
    model_module = importlib.import_module('models.' +
                                           args.model if args.model else 'models')
    model_module.update_argparser(parser)
    params = parser.parse_args()
    model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(
        params)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    input_names = ["image"]
    output_names = ["out_image"]
    dummy_input = torch.randn(1, 3, int(2040/params.scale), int(2040/params.scale))
    dynamic_axes = {'image': {0: '-1'}, 'out_image': {0: '-1'}}
    torch.onnx.export(model, dummy_input, args.output_name, input_names=input_names,
                      dynamic_axes=dynamic_axes, output_names=output_names, opset_version=9, verbose=True)

sys.path.append('./wdsr')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        help='File path to load checkpoint.',
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        help='Model name.',
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_name',
        help='Output file name.',
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        '--scale',
        help='Scale factor for image super-resolution.',
        default=2,
        type=int,
        required=True
    )
    pth2onnx(parser)
