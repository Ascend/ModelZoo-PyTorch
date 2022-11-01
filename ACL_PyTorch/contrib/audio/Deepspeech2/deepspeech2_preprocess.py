# Copyright 2021 Huawei Technologies Co., Ltd
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

"""
Data preprocessing script
"""
import os
import json
import numpy as np
import argparse
import torch
from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Deepspeech')
parser.add_argument('--data_file', default='./data/an4_test_manifest.json')
parser.add_argument('--save_path', default='./data/an4_dataset/test')
parser.add_argument('--label_file', default='./labels.json')
args = parser.parse_args()

def collate_fn(batch):
    """
    data preprocessing 
    """
    def func(p):
        """
        data size
        """
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, input_percentages, [targets, target_sizes]


if __name__ == '__main__':
    with open(to_absolute_path(args.label_file)) as label_file:
        labels = json.load(label_file)
    # if labels:
    #     print("labels ready")

    dataset = SpectrogramDataset(
        audio_conf=DataConfig.spect,
        input_path=args.data_file,
        labels=labels,
        normalize=True,
        aug_cfg=DataConfig.augmentation
    )
    inputs, input_percentages, target_list = collate_fn(dataset)
    targets = target_list[0]
    target_sizes = target_list[1]
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

    # print(inputs,input_sizes)
    if not os.path.exists(args.save_path + '/spect'): os.makedirs(args.save_path + '/spect')
    if not os.path.exists(args.save_path + '/sizes'): os.makedirs(args.save_path + '/sizes')
    i = 0
    for input_data in inputs:
        i = i + 1
        spect = np.array(input_data).astype(np.float32)
        spect.tofile(os.path.join(args.save_path + '/spect', "data" + str(i) + ".bin"))

    i = 0
    for input_size in input_sizes:
        i = i + 1
        transcript = np.array(input_size).astype(np.int32)
        transcript.tofile(os.path.join(args.save_path + '/sizes', "data" + str(i) + ".bin"))

    f = open(args.save_path + '/sizes/' + 'sizes.txt', "w")
    for w in np.array(input_sizes).astype(np.int32):
        f.write(str(w)+' ')
    f.close()

    f = open(args.save_path + '/targets.txt', "w")
    for w in np.array(targets):
        f.write(str(w) + ' ')
    f.close()

    f = open(args.save_path + '/target_sizes.txt', "w")
    for w in np.array(target_sizes).astype(np.int32):
        f.write(str(w) + ' ')
    f.close()