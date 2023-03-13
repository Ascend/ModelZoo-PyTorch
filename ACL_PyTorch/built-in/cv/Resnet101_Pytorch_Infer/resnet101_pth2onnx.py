# Copyright 2020 Huawei Technologies Co., Ltd
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
import torch.onnx
import torchvision.models as models


def pth2onnx(input_file, output_file):
    model = models.resnet101()
    checkpoint = torch.load(input_file, map_location=None)
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file, 
        input_names = input_names, 
        dynamic_axes = dynamic_axes, 
        output_names = output_names, 
        opset_version=11
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Path', add_help=False)
    parser.add_argument('--checkpoint', required=True, metavar='DIR',
                        help='path to checkpoint file')
    parser.add_argument('--save_dir', default="resnet101.onnx", type=str,
                        help='save dir for onnx model')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pth2onnx(args.checkpoint, args.save_dir)
