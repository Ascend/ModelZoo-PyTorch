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
sys.path.append('./ResNeSt')
import torch
import torch.onnx
from resnest.torch import resnest50
from collections import OrderedDict


def proc_node_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def pth2onnx(input_file, output_file):
    checkpoint = torch.load(input_file, map_location=None)
    checkpoint = proc_node_module(checkpoint)

    model = resnest50()
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.

    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      dynamic_axes=dynamic_axes,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./resnest50.pth")
    parser.add_argument('--target', type=str, default="resnest50.onnx")
    args = parser.parse_args()

    pth2onnx(args.source, args.target)
