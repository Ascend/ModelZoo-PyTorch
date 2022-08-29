# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import argparse
import numpy as np


def create_input_dataset(output_dir, data_dir):
    """ST-GCN preprocess script to create input dataset.
    Args:
        -batch_size: define batch_size of the model
        -data_dir: input data path
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    val_data_path = os.path.join(output_dir, "val_data")
    if not os.path.exists(val_data_path):
        os.mkdir(val_data_path)

    data_np= np.load(data_dir)
    sample_count = data_np.shape[0]
    for idx in range(sample_count):
        input_bin_path = os.path.join(val_data_path, "{}.bin".format(idx))
        input_data_np = np.expand_dims(data_np[idx], 0)
        print("============ create input data and label ============")
        print("[INFO] input data {}".format(idx))
        print("=====================================================")
        input_data_np = input_data_np[:, :, ::2, :, :]
        input_data_np = np.array(input_data_np)
        input_data_np.tofile(input_bin_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kinetics dataset preprocess')
    parser.add_argument('-data_dir',
        default='./data/kinetics-skeleton/val_data.npy', help='data file path')
    parser.add_argument('-output_dir',
        default='./data/kinetics-skeleton/',
        help='output the preprocessed data and label')
    args = parser.parse_args()

    # para init
    dataset_path = args.data_dir
    output_path = args.output_dir

    # create input_data
    create_input_dataset(output_path, dataset_path)