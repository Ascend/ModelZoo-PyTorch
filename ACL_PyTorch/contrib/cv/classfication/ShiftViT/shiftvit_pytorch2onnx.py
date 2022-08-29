# Copyright 2022 Huawei Technologies Co., Ltd
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


import argparse

import torch
import numpy as np
from timm.models import create_model

import models


def pytorch2onnx(ckpt_path, onnx_path, opset_version):

    model = create_model(
        'shiftvit_light_tiny',
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    device = torch.device('cpu')
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)


    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        'pytorch model convert to onnx model')
    parser.add_argument('-c', '--ckpt-path', type=str, 
                        required=True, help='path to checkpoint file')
    parser.add_argument('-o', '--onnx-path', type=str, 
                        required=True, help='path to onnx file')
    parser.add_argument('-v', '--opset-version', type=int, 
                        default=12, help='opset version')
    args = parser.parse_args()

    pytorch2onnx(args.ckpt_path, args.onnx_path, args.opset_version)

