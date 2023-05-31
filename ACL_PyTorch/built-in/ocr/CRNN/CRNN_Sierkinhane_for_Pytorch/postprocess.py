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
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from argument_parser import ArgumentParser
from lib.utils import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_config_argument()
    parser.add_preprocessed_test_label_argument()
    parser.add_predict_dir_argument()
    parser.add_is_dym()
    return parser.parse_args()


class Postprocessor:
    def __init__(self, predict_dir, label_file, alphabets, is_dym):
        self.__predict_dir = predict_dir
        self.__label_file = label_file
        self.__alphabets = alphabets
        self.__filename_to_label = self.__get_filename_to_label()
        self.dym = is_dym

    def __get_filename_to_label(self):
        lines = open(self.__label_file, encoding='utf-8').readlines()
        filename_to_label = {}
        for line in lines:
            item = line.strip('\n').split(' ')
            filename, _ = os.path.splitext(item[0])
            filename_to_label[filename] = item[1]
        return filename_to_label

    def compute_accuracy(self):
        correct_count = 0
        predict_files = os.listdir(self.__predict_dir)
        for predict_file in tqdm(predict_files):
            predict_text = self.__get_predict_text(predict_file)
            expected_text = self.__get_expected_text(predict_file)
            if self.dym:
                predict_text = set(predict_text)
                expected_text = set(expected_text)
                if expected_text.issubset(predict_text):
                    correct_count += 1
            else:
                if predict_text == expected_text:
                    correct_count += 1
        total_count = len(predict_files)
        print(f'total: {total_count}, correct_count: {correct_count}, accuracy: {correct_count / total_count}')

    def __get_expected_text(self, predict_file):
        filename, _ = os.path.splitext(predict_file)
        # ais_bench 输出的文件名格式为「原输入文件名_序号」，GitHub 提供的图片文件名带下划线
        filename = filename.rsplit('_', maxsplit=1)[0]
        expected_text = self.__filename_to_label[filename]
        return expected_text

    def __get_predict_text(self, image_file):
        filename, _ = os.path.splitext(image_file)
        predict_filepath = os.path.join(self.__predict_dir, filename + '.npy')
        predict_data = np.load(predict_filepath)
        predict_data = predict_data[:41, ...]
        predict_data = torch.from_numpy(predict_data)
        _, char_indices = predict_data.max(2)
        char_indices = char_indices.transpose(1, 0).contiguous().view(-1)
        char_indices_size = torch.autograd.Variable(torch.IntTensor([char_indices.size(0)]))
        converter = utils.strLabelConverter(self.__alphabets)
        return converter.decode(char_indices.data, char_indices_size.data, raw=False)


if __name__ == '__main__':
    args = parse_args()
    config = config.get_config(args.config)
    postprocessor = Postprocessor(args.predict_dir, args.label, config.DATASET.ALPHABETS, args.is_dym)
    postprocessor.compute_accuracy()
