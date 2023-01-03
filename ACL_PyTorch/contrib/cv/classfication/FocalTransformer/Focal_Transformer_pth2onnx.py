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
import torch.onnx
import os
import sys

def pth2onnx(code_path, input_file, output_file):
    config = _C.clone()
    _update_config_from_file(config, os.path.join(code_path,'configs/focalv2_small_useconv_patch4_window7_224.yaml'))
    model = build_model(config)
    checkpoint = torch.load(input_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn((1, 3, 224, 224))
    torch.onnx.export(model, dummy_input, output_file,
                      input_names = input_names, dynamic_axes = dynamic_axes,
                      output_names = output_names, opset_version=13, verbose=False, keep_initializers_as_inputs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type=str, default='/home/focal_transformer/')
    parser.add_argument('--input_path', type=str, default='./infer/focalv2-small-useconv-is224-ws7.pth')
    parser.add_argument('--output_path', type=str, default='./infer/focalv2-small-useconv-is224-ws7.onnx')
    args = parser.parse_args()

    sys.path.append(args.code_path)
    from config import _C, _update_config_from_file
    from classification import build_model
    pth2onnx(args.code_path, args.input_path, args.output_path)
