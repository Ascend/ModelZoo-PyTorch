# Copyright 2022 Huawei Technologies Co., Ltd
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
import glob
import json
import numpy as np
import argparse
import editdistance
from unittest import result
from pythainlp.util import normalize


def load_vocab_map(vocab_path):
    '''
    Load the token-to-index mapping dictionary and convert it to the index-to-token mapping.
    Args:
      vocab_path: filepath of vocabulary for decode
    Returns:
      index-to-token mapping dictionary
    '''
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_index = eval(f.read())
        index_token = dict((v, k) for k, v in token_index.items())
    return index_token


def remove_adjacent(item):  # code from https://stackoverflow.com/a/3460423
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return ''.join(a)


def find_files(path, pattern="*.flac"):
    '''Find files recursively according to `pattern`'''
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattern}', recursive=True):
       filenames.append(filename)
    return filenames


def post_process(model_out_dir, index_token_map, batch_i_filename_map):
    filenames = find_files(model_out_dir, "*.bin")
    results = {}
    for file in filenames:
        with open(file, 'rb') as f:
            buf = f.read()
        batch_data = np.frombuffer(buf, dtype=np.float32)
        batch_data = batch_data.reshape(-1, 312, 32)    # (batch size, sequence length, dictionary length)
        batch_i = os.path.basename(file).split('.')[0][0:-2]
        for key, data in zip (batch_i_filename_map[batch_i], batch_data):
            prediction = np.argmax(data, axis=-1)
            # Text post processing
            _t1 = ''.join([index_token_map[i] for i in list(prediction)])
            text = normalize(''.join([remove_adjacent(j) for j in _t1.split("<pad>")]))
            text = text.replace('|', ' ')
            results[key] = text

    return results


def get_id_text_dict(file_path):
    '''
    Load text file and create a mapping between sound file name and text
    '''
    dict = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            dict[line[0]] = ' '.join(line[1:])
    return dict


def eval_accuracy(ground_truth_dict, infer_dict):
    '''
    The inference data is compared with the real data to get the error rate
    Args:
      ground_truth_dict: a dictionary mapping from audio file to its corresponding ground truth text.
      infer_dict: a dictionary mapping from audio file to its corresponding infered text.
    '''
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
    parser.add_argument('--input', help='om model output',
                        default="./data/bin_om_out_bs1")
    parser.add_argument('--batch_size', help='batch size', default=1)

    args = parser.parse_args()

    base_dir = args.input
    batch_size = int(args.batch_size)

    batch_i_filename_map_path = "./data/batch_i_filename_map_bs{}.json".format(batch_size)
    with open(batch_i_filename_map_path, 'r', encoding='utf-8') as f:
        batch_i_filename_map = json.load(f)

    # load vocabulary
    vocab_path = "./wav2vec2_pytorch_model/vocab.json"
    index_token_map = load_vocab_map(vocab_path)

    # get the decoded text
    infered_result_dict = post_process(base_dir, index_token_map, batch_i_filename_map)

    ground_truth_texts_path = "./data/ground_truth_texts.txt"
    ground_truth_dict = get_id_text_dict(ground_truth_texts_path)

    # compute accuracy
    # print(infered_result_dict)
    wer = eval_accuracy(ground_truth_dict, infered_result_dict)
    
    print(f"Err: {wer}")
    print(f"Acc: {1 - wer}")
