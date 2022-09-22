# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

from my_allennlp.allennlp.modules.elmo import Elmo
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', default='elmo.onnx')
    parser.add_argument('--word_len', default=8, type=int)
    opt = parser.parse_args()
    pth2onnx(opt)


def pth2onnx(opt):
    batch_size = 1
    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1)
    elmo.eval()
    dummy_input = torch.randint(1, 10, (batch_size, opt.word_len, 50), dtype=torch.int32)
    torch.onnx.export(elmo, dummy_input, opt.output_file, input_names=["input"], 
                      output_names=["output"], opset_version=11, verbose=False)


if __name__ == '__main__':
    main()
