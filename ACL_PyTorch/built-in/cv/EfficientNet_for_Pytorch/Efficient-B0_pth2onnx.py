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
import argparse
import sys
import os
import torch
import torch.onnx
sys.path.append(r"./pycls")
from pycls import models


def main():
    parser = argparse.ArgumentParser(description="Efficient-b0 Inference")
    args = parser.parse_args()

    model = models.effnet("B0", pretrained=True)
    model.eval()
    input_names = ['image']
    dynamic_axes = {"image": {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    export_onnx_file = "Efficient-b0.onnx"
    torch.onnx.export(model, dummy_input, export_onnx_file, input_names=input_names, 
                      dynamic_axes=dynamic_axes, opset_version=11, version=True)
    

if __name__ == "__main__":
    main()
