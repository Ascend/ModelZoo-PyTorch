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

import tqdm
import argparse
from allennlp.modules.elmo import batch_to_ids
import os
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', 
                        default='1-billion-word-language-modeling-benchmark-r13output/\
                        heldout-monolingual.tokenized.shuffled/',
                        help='path to dataset')
    parser.add_argument('--save_path', default='data.txt',
                        help='preprocess file')
    parser.add_argument('--bin_path', default='bin_path',
                        help='process file')
    parser.add_argument('--file_num', default=50, type=int,
                        help='test file number')
    parser.add_argument('--word_len', default=8, type=int,
                        help='words length')

    opt = parser.parse_args()
    save_file(opt)
    process_file(opt)


def save_file(opt):
    with open(opt.save_path, 'w', encoding='utf-8') as f:
        for i in range(opt.file_num):
            if i < 10:
                fr = '{}news.en.heldout-0000{}-of-00050'.format(opt.file_path, i)
            else:
                fr = '{}news.en.heldout-000{}-of-00050'.format(opt.file_path, i)
            for line in fr.readlines():
                if len(line.strip().split()) <= opt.word_len:
                    f.write(line)


def read_file(opt):
    with open(opt.save_path, 'r', encoding='utf-8') as f:
        contexts = []
        for line in f.readlines():
            context = line.strip().split(' ')
            contexts.append(context)
        return contexts


def process_file(opt):
    if not os.path.exists(opt.bin_path):
        os.makedirs(opt.bin_path)
    contexts = read_file(opt)
    pad = torch.randint(261, 262, (1, 1, 50))
    
    pbar = tqdm.tqdm(
        total=len(contexts),
        desc='Pretprocessing',
        position=0,
        leave=True)

    for i in range(len(contexts)):
        context = [contexts[i]]
        ids = batch_to_ids(context)
        if ids.shape[1] < opt.word_len:
            gap = opt.word_len - ids.shape[1]
            for _ in range(gap):
                ids = torch.cat((ids, pad), 1)

        ids_np = np.asarray(ids, dtype='int32')
        bin_file_path = os.path.join(opt.bin_path, '{}.bin'.format(i))
        ids_np.tofile(bin_file_path)
        pbar.update(1)
    
    pbar.close()


if __name__ == '__main__':
    main()