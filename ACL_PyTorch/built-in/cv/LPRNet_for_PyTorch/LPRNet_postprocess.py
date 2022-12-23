# Copyright 2022 Huawei Technologies Co., Ltd
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
import argparse
import os
import sys

import numpy as np

sys.path.append('LPRNet_Pytorch')

from LPRNet_Pytorch.data import CHARS, CHARS_DICT

BLANK_CHAR = 67  # '-': len(CHARS) - 1
CHARS_PARSER = {i: c for i, c in enumerate(CHARS)}


def parse_name(label: list) -> str:
    """parse LPR name from label"""
    name = ""
    for num in label:
        name += CHARS_PARSER[int(num)]
    return name


def parse_result(bin_path):
    """parse inference result get label"""
    result = np.fromfile(bin_path, dtype=np.float32)  # [1224,]
    result.resize([68, 18])  # [68, 18]
    preb_label = []
    for i in range(result.shape[1]):
        preb_label.append(np.argmax(result[:, i], axis=0))

    no_repeat_blank_label = []  # remove repeat char and blank('-') char
    prev_char = preb_label[0]
    if prev_char != BLANK_CHAR:
        no_repeat_blank_label.append(prev_char)

    for char in preb_label[1:]:
        if char == BLANK_CHAR:
            prev_char = char
        elif char == prev_char:
            continue
        else:
            no_repeat_blank_label.append(char)
            prev_char = char

    return no_repeat_blank_label


def postprocess(result_path):
    """read inference result and calculate accuracy"""
    tp, tn_1, tn_2 = 0, 0, 0

    result_list = os.listdir(result_path)
    result_list = filter(lambda x: x.endswith('.bin'), result_list)

    for result_name in result_list:
        # get true label
        true_label = [CHARS_DICT[c] for c in result_name.split('_')[0]]
        # inference result label
        rst_path = os.path.join(result_path, result_name)
        preb_label = parse_result(rst_path)

        if len(preb_label) != len(true_label):
            tn_1 += 1  # length error
            print(f'[ERROR1]true content: {parse_name(true_label)}, preb content: {parse_name(preb_label)}')
            continue
        if (np.asarray(preb_label) == np.asarray(true_label)).all():
            tp += 1  # content right
            print(f'[ INFO ]true content: {parse_name(true_label)}, preb content: {parse_name(preb_label)}')
        else:
            tn_2 += 1  # content error
            print(f'[ERROR2]true content: {parse_name(true_label)}, preb content: {parse_name(preb_label)}')

    accuracy = tp / (tp + tn_1 + tn_2)
    print('=' * 70)
    print('[ INFO ]Test Accuracy: {} [{}:{}:{}]'.format(
        accuracy, tp, tn_1, tn_2, (tp + tn_1 + tn_2)))
    print("=" * 70)
    print('["ERROR1" means predict result length  is different from true content!]')
    print('["ERROR2" means predict result content is different from true content!]')
    print('=' * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', help='ais_bench result bin files path')
    args = parser.parse_args()
    postprocess(args.result_path)
