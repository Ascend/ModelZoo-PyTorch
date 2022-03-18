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
from torch.autograd import Variable
import argparse
import numpy as np
from model import Generator

def pth2onnx(input_file, output_file):
    myGenerator = Generator(64, 5, 6)
    myGenerator.load_state_dict(torch.load(input_file, map_location=lambda storage, loc: storage))
    input_names = ["real_img", "attr"]
    output_names = ["fake_img"]
    dynamic_axes = {'real_img': {0: '-1'}, 'attr': {0: '-1'}, "fake_img": {0: '-1'}}

    dummy_input1 = torch.randn(1, 3, 128, 128)
    dummy_input2 = torch.randn(1, 5)
    torch.onnx.export(myGenerator, (dummy_input1, dummy_input2), output_file, input_names = input_names,
                      output_names = output_names,dynamic_axes = dynamic_axes,opset_version=11, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    pth2onnx(args.input_file, args.output_file)