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

import torch
import torch.onnx
from collections import OrderedDict
import ssl
import torchvision.models as models

def convert():
    model = models.vgg16(pretrained=True)
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, "vgg16.onnx", input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11)

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    convert()
