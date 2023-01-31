# Copyright 2022 Huawei Technologies Co., Ltd
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

# coding: utf-8
import argparse
import time
import math
import os
import sys
import torch
from data_utils import get_lm_corpus
from utils.exp_utils import get_logger
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/enwik8',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='enwik8',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='onnx',
                    choices=['all', 'valid', 'test','onnx'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=128,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--pre_data_save_path', type=str, default='./bin_data',
                    help='location of the bin data')
parser.add_argument('--pre_target_save_path', type=str, default='./bin_target',
                    help='location of the bin data')

args = parser.parse_args()

assert args.ext_len >= 0, 'extended context length must be non-negative'

if not os.path.exists(args.pre_data_save_path):
    os.makedirs(args.pre_data_save_path)
if not os.path.exists(args.pre_target_save_path):
    os.makedirs(args.pre_target_save_path)
    
# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

valid_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                 device='cpu', ext_len=args.ext_len)
test_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                device='cpu', ext_len=args.ext_len)

f_info_file = open("bin_file.info", "wt")

for idx, (data, target, seq_len) in enumerate(valid_iter):
    if idx < valid_iter.n_batch-1:
        data_seq = np.asarray(data, dtype=np.int64)
        target_seq = np.asarray(target, dtype=np.int64)
        data_bin_file_path = os.path.join(args.pre_data_save_path, 'data_' + str(idx) + ".bin")
        target_bin_file_path = os.path.join(args.pre_target_save_path, 'data_' + str(idx) + ".bin")
        data_seq.tofile(data_bin_file_path)
        target_seq.tofile(target_bin_file_path)
        f_info_file.write(str(idx) + ' ' + args.pre_data_save_path + '/data_' + str(idx) + ".bin" + '\n')
        f_info_file.write(str(idx) + ' ' + args.pre_target_save_path + '/data_' + str(idx) + ".bin" + '\n')
        print('\rhave done {} batches'.format(str(idx+1)), end='')
    else:
        break
print('\nCompleted!')
f_info_file.close()