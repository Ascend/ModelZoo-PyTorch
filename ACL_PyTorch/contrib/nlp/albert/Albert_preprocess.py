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

from __future__ import absolute_import, division, print_function

import argparse
import os
import parse
from tqdm import tqdm
import numpy as np


def dump_input_data(save_dir, input_data, seq):
    input_names = input_data[0].keys()
    for input_name in input_names:
        sub_dir = os.path.join(save_dir, input_name)
        os.makedirs(sub_dir, exist_ok=True)

    for data_idx in tqdm(range(len(input_data))):
        data_dic = input_data[data_idx]
        for data_name, data in data_dic.items():
            data = data[:, :seq]
            save_path = os.path.join(save_dir, data_name, f"{data_idx}.npy")
            data = data.numpy()
            np.save(save_path, data)


def dump_label(save_dir, gt_label):
    gt_label = [label.numpy() for label in gt_label]
    gt_label = np.array(gt_label)
    save_path = os.path.join(save_dir, "label.npy")
    np.save(save_path, gt_label)


def om_pre(ar):
    ar.batch_size = 1
    data, _, label = parse.load_data_model(ar)
    dump_input_data(ar.save_dir, data, ar.max_seq_length)
    dump_label(ar.save_dir, label)
    print('data num: %d' % len(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_dir", type=str, default='./albert_pytorch',
                        help="prefix dir for ori model code")
    parser.add_argument("--pth_dir", type=str, default='./albert_pytorch/outputs/SST-2/',
                        help="dir of pth, load args.bin and model.bin")
    parser.add_argument("--data_dir", type=str, default='./albert_pytorch/dataset/SST-2/',
                        help="dir of dataset")
    parser.add_argument("--save_dir", type=str, default='',
                        help="save dir for preprocessed data")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="seq length for input data.")
    ar = parser.parse_args()

    ar.pth_arg_path = os.path.join(ar.pth_dir, "training_args.bin")
    ar.data_type = 'dev'
    om_pre(ar)


if __name__ == "__main__":
    main()
