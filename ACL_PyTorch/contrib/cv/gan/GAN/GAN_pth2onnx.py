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
from models import Generator
from torch.autograd import Variable
import argparse
import numpy as np
from collections import OrderedDict

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx(input_file, output_file):
    generator = Generator()
    checkpoint = torch.load(input_file, map_location=torch.device('cpu'))
    checkpoint = proc_nodes_module(checkpoint)
    generator.load_state_dict(checkpoint)
    input_names = ["Z"]
    output_names = ["generateimg"]
    dynamic_axes = {'Z': {0: '-1'}, 'generateimg': {0: '-1'}}

    Tensor = torch.FloatTensor
    dummy_input = Variable(Tensor(np.random.normal(0, 1, (16, 100))))
    torch.onnx.export(generator, dummy_input, output_file, input_names = input_names,
                      output_names = output_names,dynamic_axes = dynamic_axes,opset_version=11, verbose=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    pth2onnx(args.input_file, args.output_file)
