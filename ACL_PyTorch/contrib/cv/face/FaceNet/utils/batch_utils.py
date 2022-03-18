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


import numpy as np
import os
import argparse


def np2bin(save_np, save_name, save_path):
    save_np.tofile(save_path + '/' + save_name + '_output_0.bin')


def bin2np(bin_path):
    return np.fromfile(bin_path, dtype=np.float32)


def general_data(batch_size, data_root_path, save_root_path):
    in_files = os.listdir(data_root_path)
    for file_name in in_files:
        file_index = file_name.split('_')[0]
        bin_file = bin2np(data_root_path + '/' + file_name)
        img_n = bin_file.shape[0] // 512
        bin_file = bin_file.reshape([img_n, 512])
        file_index_i = int(file_index)
        for i in range(img_n):
            if file_index_i * batch_size + i < 13233:
                np2bin(bin_file[i], str(file_index_i * batch_size + i), save_root_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--data_root_path', type=str, help='data path')
    parser.add_argument('--save_root_path', type=str, help='save path')
    arg = parser.parse_args()
    general_data(arg.batch_size, arg.data_root_path, arg.save_root_path)
