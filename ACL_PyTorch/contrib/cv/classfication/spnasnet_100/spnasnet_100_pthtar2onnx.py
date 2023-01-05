# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import sys
import timm
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
import torch.onnx

from collections import OrderedDict


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(input_file, output_file):
    checkpoint = torch.load(input_file, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = timm.create_model('spnasnet_100')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, verbose=True, opset_version=11)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert(input_file, output_file)
