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
import torch
import torch.onnx
import torch.nn as nn
sys.path.append(r"./Efficient-3DCNNs")
from models import mobilenetv2

def convert(pth_file_path, onnx_file_path):
    model = mobilenetv2.get_model(
        num_classes=101,
        sample_size=112,
        width_mult=1.0)
    model.classifier = nn.Sequential(
        nn.Dropout(0.9),
        nn.Linear(model.classifier[1].in_features, 101)
    )
    checkpoint = torch.load(pth_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print(model)

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])