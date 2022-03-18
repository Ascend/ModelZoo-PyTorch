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

"""
use 'pth' model file to generate result from 'prep_data' dataset on cpu.
"""

from collections import OrderedDict
import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(r"PyTorch-GAN/implementations/dcgan")
from dcgan import Generator


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def main(checkpoint_path, dataset_path, save_path):
    shape = (1, 100, 1, 1)
    model = Generator(32, 100, 1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['G']
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    input_name_list = os.listdir(dataset_path)
    input_name_list.sort()
    for name in input_name_list:
        input_path = os.path.join(dataset_path, name)
        output_path = os.path.join(save_path, name[:-4] + "_1.bin")
        print(input_path, " ===> ", output_path)
        inp = np.fromfile(input_path, dtype=np.float32)
        inp = inp.reshape(shape)
        inp = torch.from_numpy(inp)
        img = model(inp)
        img = img.detach().numpy()
        img.tofile(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="./checkpoint-amp-epoch_200.pth", type=str)
    parser.add_argument("--dataset_path", default="./prep_dataset/", type=str)
    parser.add_argument("--save_path", default="./pth_result/", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args.checkpoint_path, args.dataset_path, args.save_path)
