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
import argparse
import torch
import torch.onnx
sys.path.append(r"./pretrained-models.pytorch")
import pretrainedmodels

def pth2onnx(input_file, output_file):
    model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=None)
    checkpoint = torch.load(input_file, map_location=None)
    model.load_state_dict(checkpoint)

    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=False, opset_version=11)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SENet pth2onnx')
    parser.add_argument("--pth", default="./se_resnet50-ce0d4300.pth", help='pth file')
    parser.add_argument("--onnx", default="./se_resnet50_dybs.onnx", help='save onnx file')
    parsers = parser.parse_args()
    
    pth2onnx(parsers.pth, parsers.onnx)
    
    print("Successfully exported ONNX model: {}".format(parsers.onnx))
