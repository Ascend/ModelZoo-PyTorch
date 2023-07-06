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


import ssl
import sys

import torch
import torchvision.models as models


def convert(checkpoint=None, output_file='./inceptionv3.onnx'):
    if (checkpoint):
        model = models.inception_v3(
            pretrained=False, 
            transform_input=False, 
            init_weights=False
        )
        checkpoint = torch.load(checkpoint, map_location=None)
        model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()})
    else:
        model = models.inception_v3(pretrained=True)

    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}

    dummy_input = torch.randn(1, 3, 299, 299)
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file, 
        input_names=input_names, 
        output_names=output_names, 
        dynamic_axes=dynamic_axes,
        opset_version=11
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('data preprocess.')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='path to PyTorch pretrained file(.pth)')
    parser.add_argument('--onnx', type=str, default='./inceptionv3.onnx', 
                        help='path to save onnx model(.onnx)')
    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context
    convert(args.checkpoint, args.onnx)
