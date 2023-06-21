# Copyright 2023 Huawei Technologies Co., Ltd
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


def convert(pthfile):
    model = models.resnet50(pretrained=False)
    resnet50 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(resnet50)

    input_names = ["input"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input,
        "resnet50_official.onnx",
        input_names=input_names, 
        output_names=output_names, 
        opset_version=11)


if __name__ == "__main__":
    pth_path = sys.argv[1]
    convert(pth_path)
