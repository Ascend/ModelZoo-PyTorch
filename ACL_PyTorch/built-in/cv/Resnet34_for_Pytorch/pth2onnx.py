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

import sys
import torch
import torch.onnx
import torchvision.models as models


def convert(input_model, output_model):
    model = models.resnet34(pretrained=False)
    pthfile = input_model
    resnet34 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(resnet34)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_model,
        input_names = input_names,
        dynamic_axes = dynamic_axes,
        output_names = output_names,
        opset_version = 11)

if __name__ == "__main__":
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    convert(input_model, output_model)

