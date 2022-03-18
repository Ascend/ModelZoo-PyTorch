# ============================================================================
# Copyright 2018-2019 Open-MMLab. All rights reserved.
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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

import os
import pickle as pk
import argparse
import torch
import numpy as np
from mmskeleton.utils import call_obj


def create_input_dataset(output_dir, data_dir, label_dir):
    """ST-GCN preprocess script to create input dataset.
    Args:
        -batch_size: define batch_size of the model
        -data_dir: input data path
        -label_dir: input label path
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    val_data_path = os.path.join(output_dir, "val_data")
    val_label_path = os.path.join(output_dir, "val_label")
    if not os.path.exists(val_data_path):
        os.mkdir(val_data_path)
    if not os.path.exists(val_label_path):
        os.mkdir(val_label_path)

    data_np= np.load(data_dir)
    with open(label_dir, "rb") as f:
        labels_file = pk.load(f)
        labels = labels_file[1]
    sample_count = data_np.shape[0]
    for idx in range(sample_count):
        input_bin_path = os.path.join(val_data_path, "{}.bin".format(idx))
        label_bin_path = os.path.join(val_label_path, "{}.bin".format(idx))
        input_data_np = np.expand_dims(data_np[idx], 0)
        label_data_np = np.expand_dims(labels[idx], 0)
        label_data_np = np.array(label_data_np, dtype=np.int32)
        print("============ create input data and label ============")
        print("[INFO] input data {}".format(idx))
        print("[INFO] label is {}".format(labels[idx]))
        print("=====================================================")
        input_data_np = np.array(input_data_np)
        input_data_np.tofile(input_bin_path)
        label_data = np.array(label_data_np)
        label_data.tofile(label_bin_path)


def preprocess(dataset_cfg, output_dir, batch_size=1, workers=1):
    """ST-GCN preprocess script to create input dataset.
    Args:
        -dataset_cfg: dataset config file
        -output_dir: ouput preprocessed data files
        -batch_size: define batch_size of the model
    """
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=workers)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    val_data_path = os.path.join(output_dir, "val_data")
    val_label_path = os.path.join(output_dir, "val_label")
    if not os.path.exists(val_data_path):
        os.mkdir(val_data_path)
    if not os.path.exists(val_label_path):
        os.mkdir(val_label_path)

    idx = 0
    for data, label in data_loader:
        print("[INFO] process current data ID {}".format(idx))
        val_data_file = os.path.join(val_data_path, "{}.bin".format(idx))
        val_label_file = os.path.join(val_label_path, "{}.bin".format(idx))
        val_data = np.array(data)
        val_label = np.array(label)
        val_data.tofile(val_data_file)
        val_label.tofile(val_label_file)
        idx = idx + 1
    print("[INFO] process data finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kinetics dataset preprocess')
    parser.add_argument('-data_dir',
        default='./data/kinetics-skeleton/val_data.npy', help='data file path')
    parser.add_argument('-label_dir',
        default='./data/kinetics-skeleton/val_label.pkl',
        help='label file path')
    parser.add_argument('-output_dir',
        default='./data/kinetics-skeleton/',
        help='output the preprocessed data and label')
    parser.add_argument('-batch_size', default=1,
        help='batch size of the model input')
    parser.add_argument('-num_workers', default=1,
        help='count of the dataloader')
    args = parser.parse_args()

    # para init
    batch = args.batch_size
    dataset_path = args.data_dir
    labels_path = args.label_dir
    num_worker = args.num_workers
    output_path = args.output_dir

    dataset_file = {
        'type': 'deprecated.datasets.skeleton_feeder.SkeletonFeeder',
        'data_path': dataset_path,
        'label_path': labels_path
    }

    # create input_data
    create_input_dataset(output_path, dataset_path, labels_path)

    # preprocess
    preprocess(dataset_file, output_path, int(batch), int(num_worker))