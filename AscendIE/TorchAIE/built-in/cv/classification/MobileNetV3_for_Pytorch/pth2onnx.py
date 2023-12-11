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

import os
import argparse
from collections import OrderedDict

import torch
import onnx
import onnxsim

from mobilenetv3 import MobileNetV3_Small

def adjust_checkpoint(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key == "module.features.0.0.weight":
            print(value)
        if key[0:7] == "module.":
            name = key[7:]
        else:
            name = key[0:]
        
        new_state_dict[name] = value
    return new_state_dict


def get_model(args):
    checkpoint = torch.load(args.pth_model, map_location='cpu')['state_dict']
    checkpoint = adjust_checkpoint(checkpoint)
    model_pt = MobileNetV3_Small()
    model_pt.load_state_dict(checkpoint)
    model_pt.eval()
    return model_pt

def pth2onnx(args, model_pt):
    input_names = ["input"]
    output_names = ["output"]
    input_data = torch.randn(args.batch_size, 3, 224, 224)
    output_onnx = os.path.join(args.output_dir, args.onnx_model)
    torch.onnx.export(
        model_pt, input_data, output_onnx,
        input_names=input_names, output_names=output_names,
        opset_version=args.opset, export_params=True, verbose=False, do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if args.dynamic else None)
    
    # Checks
    model_onnx = onnx.load(output_onnx)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if args.simplify:
        try:
            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                test_input_shapes={'input': list(input_data.shape)} if args.dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, output_onnx)
        except Exception as e:
            print(f'simplifier failure: {e}')
    
    return model_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export MobileNetV3 onnx")
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--pth-model', type=str, default='model.pt')
    parser.add_argument('--onnx-model', type=str, default='model.onnx')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    args = parser.parse_args()
    
    ### load model
    model_pt = get_model(args)
 
    ### pth2onnx
    model_onnx = pth2onnx(args, model_pt)
    print("导出onnx success")
