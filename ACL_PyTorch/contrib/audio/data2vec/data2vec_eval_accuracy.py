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

from logging import exception
import editdistance
import argparse


def get_id_text_dict(file_path):
    dict = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            dict[line[0]] = ''.join(line[1:])
    return dict


def eval_accurary(ground_truth_dict, infer_dict):
    totol_char = 0
    edit_distance = 0
    for key, text in infer_dict.items():
        try:
            ground_truth_text = ground_truth_dict[key]
        except Exception:
            print("skip error key: ", key)
            continue

        totol_char += len(ground_truth_text)
        edit_distance += editdistance.distance(text, ground_truth_text)
    return edit_distance/totol_char


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_text', help='ground truth text', default="./data/ground_truth_texts.txt")
    parser.add_argument('--infered_text', help='infered text', default="./data/infered_texts_bs1.txt")

    args = parser.parse_args()

    ground_truth_texts_path = args.ground_truth_text
    infered_texts_path = args.infered_text
    g_dict = get_id_text_dict(ground_truth_texts_path)
    i_dict = get_id_text_dict(infered_texts_path)
    print("ground truth items: ", len(g_dict))
    print("infered items:      ", len(i_dict))
    wer = eval_accurary(g_dict, i_dict)
    print("wer: ", wer)