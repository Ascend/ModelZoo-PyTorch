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

import sys

import torch
import torchvision

def pth2onnx(input_file, output_file):
    model = torchvision.models.shufflenet_v2_x1_0()     
    model.load_state_dict(torch.load(input_file, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ['input_0']
    output_names = ['output_0']
    dynamic_axes = {'input_0':{0:"-1"}, 'output_0':{0:"-1"}}

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
        opset_version=11
    )

if __name__=="__main__":
    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]
    pth2onnx(pth_file, onnx_file)