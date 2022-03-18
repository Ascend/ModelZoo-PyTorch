# Copyright 2020 Huawei Technologies Co., Ltd
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

import sys 
import torch
import torch.onnx


sys.path.append('.')

import senet 


device = 'cpu'


def convert(file_path):
    model = senet.senet154(num_classes=1000, pretrained='imagenet', use_pretrained=False).to(device)
    state_dict = torch.load(file_path, map_location=device)["net"]
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state_dict.items()})
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "senet154.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert(sys.argv[1])