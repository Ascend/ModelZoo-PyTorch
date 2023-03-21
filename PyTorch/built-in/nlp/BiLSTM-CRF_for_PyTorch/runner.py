# coding: UTF-8
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

import argparse
import os
import ast
import torch
if torch.__version__ >= "1.8":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

from utils import extend_maps
from bugfix import bilstm_train_and_eval
from bugfix import build_corpus
from models.config import TrainingConfig

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '26118'
os.environ['WORLD_SIZE'] = '8'


def boolean_string(s: str):
    if s.upper() not in ["FALSE", "TRUE"]:
        raise ValueError(f'{s} not a valid boolean string')
    return s.upper() == "TRUE"


def parse_option():
    parser = argparse.ArgumentParser('NER script', add_help=False)
    # easy config modification
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument("--bin_mode", type=boolean_string, default=False, help="enable bin conmpile")
    parser.add_argument("--train_epochs", type=int, help='train epoch num')
    parser.add_argument("--print_step", type=int, help='step print interval')
    parser.add_argument("--batch_size", type=int, help='train batch size')
    parser.add_argument("--profiling", type=str, help='profiling type')
    parser.add_argument("--p_start_step", type=int, default=-1, help='profiling start step')
    parser.add_argument("--iteration_num", type=int, default=-1, help='train iteration number')
    parser.add_argument('--ND', type=ast.literal_eval, default=False, help="enable nd compile")
    args = parser.parse_args()

    return args


def main():
    if args.ND:
        print('***********allow_internal_format = False*******************')
        torch.npu.config.allow_internal_format = False
    else:
        torch.npu.config.allow_internal_format = True
    if args.bin_mode:
        print('bin_mode is on')
        torch.npu.set_compile_mode(jit_compile=False)
    if args.train_epochs:
        TrainingConfig.epoches = args.train_epochs
    if args.batch_size:
        TrainingConfig.batch_size = args.batch_size
    if args.print_step:
        TrainingConfig.print_step = args.print_step
    # 读取数据
    print("Reading Data...")
    if args.data_path:
        train_word_lists, train_tag_lists, word2id, tag2id = \
            build_corpus("train", data_dir=args.data_path)
        dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir=args.data_path)
        test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=args.data_path)
    else:
        train_word_lists, train_tag_lists, word2id, tag2id = \
            build_corpus("train")
        dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
        test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估BI-LSTM模型
    print("Training to evaluate BiLSTM model...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False,
        args=args
    )


if __name__ == "__main__":
    args = parse_option()
    main()