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
import torch
import torch.onnx
import sys
sys.path.append('./PraNet')
from collections import OrderedDict
from lib.PraNet_Res2Net import PraNet



def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def convert(pth_file_path, onnx_file_path):
    model = PraNet()
    pretrained_dict = torch.load(pth_file_path, map_location="cpu")
    model.load_state_dict({k.replace('module.',''):v for k, v in pretrained_dict.items()})
    if "fc.weight" in pretrained_dict:
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
    model.load_state_dict(pretrained_dict)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 352, 352)
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names,
                      opset_version=11)
if __name__ == "__main__":
    pth_path = sys.argv[1]
    onnx_path = sys.argv[2]
    convert(pth_path, onnx_path)
