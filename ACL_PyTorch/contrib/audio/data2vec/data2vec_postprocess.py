# Copyright 2023 Huawei Technologies Co., Ltd
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


def post_process(model_out_dir, index_token_map, filename_map):
    filenames = find_files(model_out_dir, "*.bin")
    results = []
    for file in filenames:
        data = np.fromfile(file, dtype=np.float32)
        data = data.reshape(1747, 32)    # (批大小，字符的总个数，字符的类型总数)
        i = os.path.basename(file).split('.')[0].split('_')[0]
        prediction = np.argmax(data, axis=-1)
        # Text post processing
        _t1 = ''.join([index_token_map[i] for i in list(prediction)])
        text = normalize(''.join([remove_adjacent(j) for j in _t1.split("<pad>")]))
        text = text.replace('|', ' ')
        results.append(filename_map[i] + ' ' + text)

    infered_texts_path = "./data/infered_texts.txt"
    with open(infered_texts_path, mode='w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='om model output',
                        default="data/bin_om_out_bs1")

    args = parser.parse_args()

    base_dir = args.input

    filename_map_path = "data/filename_map.json"
    with open(filename_map_path, 'r', encoding='utf-8') as f:
        filename_map = json.load(f)

    vocab_path = "./data2vec_pytorch_model/vocab.json"
    index_token_map = load_vocab_map(vocab_path)

    post_process(base_dir, index_token_map, filename_map)