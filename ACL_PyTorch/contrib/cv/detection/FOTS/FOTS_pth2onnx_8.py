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
from model import FOTSModel

def pth2onnx(input_file, output_file):

    model = FOTSModel()
    checkpoint = torch.load(input_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    input_names = ["image"]
    output_names = ["location"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(8, 3, 1248, 2240)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=True, opset_version=11)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pth2onnx(input_file, output_file)