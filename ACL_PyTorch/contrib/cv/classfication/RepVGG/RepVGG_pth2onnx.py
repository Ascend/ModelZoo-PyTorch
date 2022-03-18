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
from collections import OrderedDict
sys.path.append(r"./RepVGG")
import torch
import torch.onnx
from repvgg import get_RepVGG_func_by_name


def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model


def convert():
    repvgg_build_func = get_RepVGG_func_by_name("RepVGG-A0")
    model = repvgg_build_func(deploy=False)
    model = load_my_state_dict(model, torch.load(input_file, map_location='cpu'))
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {"actual_input_1" : {0 : "-1"}, "output1" : {0 : "-1"}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                    output_names=output_names, opset_version=11)


if __name__ == "__main__":
    input_file = sys.argv[1] # "RepVGG-A0-train.pth"
    output_file = sys.argv[2] # "RepVGG.onnx"
    convert()
