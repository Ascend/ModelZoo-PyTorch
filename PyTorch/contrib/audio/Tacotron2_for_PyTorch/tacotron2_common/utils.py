# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
import json
import os

import numpy as np
import torch
from scipy.io.wavfile import read


class ParseFromConfigFile(argparse.Action):

    def __init__(self, option_strings, config_type, dest, config_help=None, required=False):
        super(ParseFromConfigFile, self).__init__(option_strings=option_strings, type=config_type, dest=dest,
                                                  help=config_help, required=required)

    def __call__(self, parser, namespace, values, option_string):
        with open(values, 'r') as f:
            data = json.load(f)

        for group in data.keys():
            for k, v in data[group].items():
                underscore_k = k.replace('-', '_')
                setattr(namespace, underscore_k, v)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    max_len = 192
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = mask <= 0
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            return path, text

        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.npu.is_available():
        x = x.npu(non_blocking=True)
    return x
