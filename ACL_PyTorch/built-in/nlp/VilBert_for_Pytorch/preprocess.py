# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#


import os
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from allennlp.common.plugins import import_plugins
from allennlp.models.archival import load_archive
from allennlp.data import DataLoader


INPUT_NAMES = ['box_features', 'box_coordinates', 'box_mask',
               'token_ids', 'mask', 'type_ids', 'labels', 'label_weights']
SEQ_NAMES = ["token_ids", "mask", "type_ids"]
FEATURE_NAMES = ["box_features", "box_coordinates", "box_mask"]


def build_dataset_reader(archive_file, dataset_name='balanced_real_val', batch_size=1):
    # Load archive
    archive = load_archive(
        archive_file,
        weights_file=None,
        cuda_device=-1,
        overrides=''
    )
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader
    config = deepcopy(archive.config)
    data_loader_params = config.get("data_loader")
    data_loader_params["batch_size"] = batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=dataset_name
    )
    data_loader.index_with(model.vocab)
    return data_loader


def save_data(batch_data, data_idx, save_dir):
    def dump_data(key_name):
        os.makedirs(os.path.join(save_dir, f"{key_name}"), exist_ok=True)
        np.save(os.path.join(save_dir, f"{key_name}/{data_idx}.npy"),
                batch_data[key_name])

    for input_name in batch_data:
        dump_data(input_name)


def pad_data(batch_data, pad_len, box_num):
    out_data = {}
    for input_name in INPUT_NAMES:
        if input_name in SEQ_NAMES:
            out_data[input_name] = batch_data['question']['tokens'][input_name].numpy()
        else:
            out_data[input_name] = batch_data[input_name].numpy()
        last_dim = out_data[input_name].shape[-1]
        input_dtype = out_data[input_name].dtype

        if pad_len != -1 and input_name in SEQ_NAMES:
            padded_data = np.zeros([1, pad_len], dtype=input_dtype)
            if last_dim > pad_len:
                padded_data = out_data[input_name][:, :pad_len]
            else:
                padded_data[:, :last_dim] = out_data[input_name]
            out_data[input_name] = padded_data
        elif box_num != -1 and input_name in FEATURE_NAMES:
            input_shape = out_data[input_name].shape
            if len(input_shape) == 3:
                mid_dim = out_data[input_name].shape[-2]
                padded_data = np.zeros([1, box_num, last_dim], dtype=input_dtype)
                if mid_dim > box_num:
                    padded_data = out_data[input_name][:, :box_num, :]
                else:
                    padded_data[:, :mid_dim, :] = out_data[input_name]
            elif len(input_shape) == 2:
                padded_data = np.zeros([1, box_num], dtype=input_dtype)
                if last_dim > box_num:
                    padded_data = out_data[input_name][:, :box_num]
                else:
                    padded_data[:, :last_dim] = out_data[input_name]
            out_data[input_name] = padded_data
    return out_data


def preprocess(data_loader, save_dir, pad_len, box_num):
    for input_name in INPUT_NAMES:
        os.makedirs(os.path.join(save_dir, input_name),
                    exist_ok=True)
        for data_idx, batch_data in enumerate(iter(data_loader)):
            batch_data = pad_data(batch_data, pad_len, box_num)
            save_data(batch_data, data_idx, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess for vilbert')
    parser.add_argument('--archive_file', help='path for archive file.')
    parser.add_argument('--save_dir', help='save dir for preprocessed data.')
    parser.add_argument('--pad_len', type=int, default=-1, help='padding sequence to fixed length.')
    parser.add_argument('--box_num', type=int, default=-1, help='box num to fixed length.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    import_plugins()
    data_loader = build_dataset_reader(
        args.archive_file
    )
    preprocess(data_loader, args.save_dir, args.pad_len, args.box_num)
