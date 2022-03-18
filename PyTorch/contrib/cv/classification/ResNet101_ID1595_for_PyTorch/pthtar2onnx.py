# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torchvision
import torch.onnx

def convert():
    checkpoint = torch.load("./checkpoint.pth.tar", map_location='cpu')
    model = torchvision.models.resnet101(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], False)
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "resnet101_npu_16.onnx", 
                      input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert()
