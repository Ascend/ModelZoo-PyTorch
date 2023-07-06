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

from collections import OrderedDict
import sys

import onnx
import torch

import crnn


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(path, out):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = crnn.CRNN(32, 1, 37, 256)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 1, 32, 100)
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {1: '-1'}}
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model,
                      dummy_input,
                      out,
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
