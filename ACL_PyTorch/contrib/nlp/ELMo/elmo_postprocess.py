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
import numpy as np


def cosine_similarity(x, y):
    x1 = x.flatten().astype(dtype='float64')
    y1 = y.flatten().astype(dtype='float64')
    dot = np.dot(x1, y1)
    lx = np.linalg.norm(x1)
    ly = np.linalg.norm(y1)
    cos = dot / (lx * ly)
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='bin_path/')
    parser.add_argument('--om_output')
    parser.add_argument('--option_file', default="elmo_2x4096_512_2048cnn_2xhighway_options.json")
    parser.add_argument('--weight_file', default="elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
    opt = parser.parse_args()
    
    elmo = Elmo(opt.option_file, opt.weight_file, 1)
    elmo.eval()
    
    om_output_path = opt.om_output

    similarity = 0
    nums = len(os.listdir(opt.inputs))
    for i in range(nums):
        input_file = np.fromfile(opt.inputs + '{0}.bin'.format(i), dtype='int32').reshape((1, 8, 50))
        input_file = torch.from_numpy(input_file)
        
        om_output_file = np.fromfile(om_output_path + '{0}_0.bin'.format(i), dtype='float32').reshape((1, 8, 1024))
        
        output = elmo.forward(input_file)
        
        output = output['elmo_representations'][0].detach().numpy()

        cosine_sim = cosine_similarity(om_output_file, output)
        similarity += cosine_sim
    print('average similarity: ', similarity / nums)


if __name__ == '__main__':
    main()
