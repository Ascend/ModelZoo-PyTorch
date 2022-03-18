# Copyright 2021 Huawei Technologies Co., Ltd
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

# coding: UTF-8

import os
import argparse
import pickle as pkl
import numpy as np


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl')
parser.add_argument('--dataset', type=str, default='./Chinese-Text-Classification-Pytorch/THUCNews')
parser.add_argument('--pad_size', type=int, default=32)
parser.add_argument('--train_path', type=str, default='data/train.txt')
parser.add_argument('--test_path', type=str, default='data/test.txt')
parser.add_argument('--save_folder', type=str, default='')
args = parser.parse_args()

args.test_path = os.path.join(args.dataset, args.test_path)
args.train_path = os.path.join(args.dataset, args.train_path)
args.vocab_path = os.path.join(args.dataset, args.vocab_path)
if args.save_folder == '':
    args.save_folder = args.dataset + '_bin'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer_, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f_:
        for line_ in f_:
            lin = line_.strip()
            if not lin:
                continue
            content_ = lin.split('\t')[0]
            for word_ in tokenizer_(content_):
                vocab_dic[word_] = vocab_dic.get(word_, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
        vocab_list = vocab_list[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


if __name__ == '__main__':

    """
    Usage:
    python preprocess_to_bin.py
    """

    if args.word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(args.vocab_path):
        vocab = pkl.load(open(args.vocab_path, 'rb'))
    else:
        assert args.train_path != ''
        vocab = build_vocab(args.train_path, tokenizer_=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(args.vocab_path, 'wb+'))
    print(f"Vocab size: {len(vocab)}")
    print('bin file save path: ', os.path.abspath(args.save_folder))

    contents = []
    f = open(args.test_path, 'r', encoding='UTF-8')
    idx = 0
    for line in f:
        lin = line.strip()
        if not lin:
            continue
        content, label = lin.split('\t')
        words_line = []
        token = tokenizer(content)
        if args.pad_size:
            if len(token) < args.pad_size:
                token.extend([PAD] * (args.pad_size - len(token)))
            else:
                token = token[:args.pad_size]
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        # convert to bin
        words_line_np = np.asarray(words_line, dtype=np.int64)
        bin_file_path = os.path.join(args.save_folder, '{}_{}.bin'.format(idx, label))
        words_line_np.tofile(bin_file_path)
        idx += 1

    f.close()