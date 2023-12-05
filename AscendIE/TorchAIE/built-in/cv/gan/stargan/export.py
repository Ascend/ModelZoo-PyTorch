# Copyright 2023 Huawei Technologies Co., Ltd
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


import argparse

import torch

from model import Generator


def pth2ts(input_file):
    myGenerator = Generator(64, 5, 6)
    myGenerator.load_state_dict(torch.load(input_file, map_location=lambda storage, loc: storage))
    
    dummy_input1 = torch.randn(1, 3, 128, 128)
    dummy_input2 = torch.randn(1, 5)
    ts_model = torch.jit.trace(myGenerator, (dummy_input1, dummy_input2))
    ts_model.save("./stargan.ts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="/onnx/stargan/200000-G.pth")
    args = parser.parse_args()

    pth2ts(args.input_file)