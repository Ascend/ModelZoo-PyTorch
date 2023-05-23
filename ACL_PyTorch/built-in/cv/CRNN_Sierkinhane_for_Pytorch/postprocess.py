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

# coding=utf-8

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import config
from default_arguments import CONFIG_FILE, PREPROCESSED_LABEL_FILE
from lib.utils import utils


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_FILE)
    parser.add_argument('--label', type=str, default=PREPROCESSED_LABEL_FILE)
    parser.add_argument('--predict-dir', type=str)
    return parser.parse_args()


def compute_accuracy(alphabets, label_file, predict_dir):
    expected_text = __get_expected_text(label_file)
    predict_text = __get_predict_text(alphabets, predict_dir)
    correct_count = 0
    for expected, predict in zip(expected_text, predict_text):
        if expected == predict:
            correct_count += 1
    print(f'total: {len(predict_text)}, correct_count: {correct_count}, accuracy: {correct_count / len(predict_text)}')


def __get_expected_text(label_file):
    expected_lines = open(label_file, encoding='utf-8').readlines()
    return [line.strip('\n').split(' ')[-1] for line in expected_lines]


def __get_predict_text(alphabets, predict_dir):
    predict_files = os.listdir(predict_dir)
    predict_files.sort()
    predict_text = []
    for predict_file in tqdm(predict_files):
        predict_filepath = os.path.join(predict_dir, predict_file)
        predict_text.append(__get_predict_text_from_single_image(alphabets, predict_filepath))
    return predict_text


def __get_predict_text_from_single_image(alphabets, predict_filepath):
    predict_data = np.load(predict_filepath, dtype=np.float32)
    predict_data = torch.from_numpy(predict_data)
    _, char_indices = predict_data.max(2)
    char_indices = char_indices.transpose(1, 0).contiguous().view(-1)
    char_indices_size = torch.autograd.Variable(torch.IntTensor([char_indices.size(0)]))
    converter = utils.strLabelConverter(alphabets)
    return converter.decode(char_indices.data, char_indices_size.data, raw=False)


if __name__ == '__main__':
    args = parse_arg()
    config = config.get_config(args.config)
    compute_accuracy(config.DATASET.ALPHABETS, args.label, args.predict_dir)
