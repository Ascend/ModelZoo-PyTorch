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

# coding=UTF-8

import sys
import torch
import onnx
from pretrainedmodels.models.senet import se_resnet50


def pth2onnx(pth_file, onnx_file):
    model = se_resnet50(num_classes=1000, pretrained=None)
    model.load_state_dict(torch.load(pth_file))
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("usage: python SE_ResNet50_pth2onnx.py <pth_file> <onnx_file>")
    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]
    pth2onnx(pth_file, onnx_file)
