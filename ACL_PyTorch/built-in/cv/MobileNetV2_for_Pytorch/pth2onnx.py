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
import torchvision.models as models


def convert():
    model = models.mobilenet_v2(pretrained=False)
    pthfile = './mobilenet_v2-b0353104.pth'
    mobilenet_v2 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(mobilenet_v2)
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    dynamic_axes = {"actual_input_1": {0: "-1"}, "output1": {0: "-1"}}
    torch.onnx.export(
        model, 
        dummy_input,
        "./output/mobilenet_v2.onnx",
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11)


if __name__ == "__main__":
    convert()

