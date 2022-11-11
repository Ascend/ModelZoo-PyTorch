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

import json
import os
import glob
import json
from unittest import result
import numpy as np
import argparse
from pythainlp.util import normalize


def load_vocab_map(vocab_path):
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
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattern}', recursive=True):
       filenames.append(filename)
    return filenames


def post_process(model_out_dir, index_token_map, batch_i_filename_map):
    filenames = find_files(model_out_dir, "*.bin")
    results = []
    for file in filenames:
        with open(file, 'rb') as f:
            buf = f.read()
        batch_data = np.frombuffer(buf, dtype=np.float32)
        batch_data = batch_data.reshape(-1, 1747, 32)    # (批大小，字符的总个数，字符的类型总数)
        print(os.path.basename(file))
        batch_i = os.path.basename(file).split('_')[0:2]
        batch_i = "_".join(batch_i)
        for key, data in zip (batch_i_filename_map[batch_i], batch_data):
            print(key)
            prediction = np.argmax(data, axis=-1)
            # Text post processing
            _t1 = ''.join([index_token_map[i] for i in list(prediction)])
            text = normalize(''.join([remove_adjacent(j) for j in _t1.split("<pad>")]))
            text = text.replace('|', ' ')
            results.append(key + ' ' + text)

    infered_texts_path = "./data/infered_texts_bs{}.txt".format(len(batch_i_filename_map['batch_0']))
    with open(infered_texts_path, mode='w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='om model output',
                        default="data/bin_om_out_bs1")
    parser.add_argument('--batch_size', help='batch size', default=1)

    args = parser.parse_args()

    base_dir = args.input
    batch_size = int(args.batch_size)

    batch_i_filename_map_path = "data/batch_i_filename_map_bs{}.json".format(batch_size)
    with open(batch_i_filename_map_path, 'r', encoding='utf-8') as f:
        batch_i_filename_map = json.load(f)

    vocab_path = "./data2vec_pytorch_model/vocab.json"
    index_token_map = load_vocab_map(vocab_path)

    post_process(base_dir, index_token_map, batch_i_filename_map)