#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
pytorch-dl
Created by raj at 11:05 
Date: January 18, 2020	
"""
import sys
from argparse import ArgumentParser

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader_bert import BertDataSet
from dataset.vocab import WordVocab
from models.bert import Bert
from models.bert_lm import BertLanguageModel
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def go(arg):
    with open(arg.path) as f:
        vocab = WordVocab(f)
        vocab.save_vocab("sample-data/vocab.pkl")

    vocab = WordVocab.load_vocab("../sample-data/vocab.pkl")
    data_set = BertDataSet(arg.path, vocab, arg.max_length)

    lr_warmup = arg.lr_warmup
    batch_size = arg.batch_size

    data_loader = DataLoader(data_set, batch_size=batch_size)

    vocab_size = len(vocab.stoi)
    bert = Bert(vocab_size, width=arg.dim_model, depth=arg.depth, heads=arg.num_heads)
    model = BertLanguageModel(bert, vocab_size)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0, amsgrad=False)
    lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    cuda_condition = torch.npu.is_available()
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    if cuda_condition:
        model.npu()

    if cuda_condition and torch.npu.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.npu.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2,3])

    for epoch in range(arg.num_epochs):
        avg_loss = 0
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Running epoch: {}".format(epoch),
                              total=len(data_loader))
        for i, data in data_iter:
            data = {key: value.to(f'npu:{NPU_CALCULATE_DEVICE}') for key, value in data.items()}
            bert_input, bert_label, segment_label, is_next = data
            bert_out, sentence_pred = model(data[bert_input], data[segment_label])
            mask_loss = criterion(bert_out.transpose(1, 2), data[bert_label])
            next_loss = criterion(sentence_pred, data[is_next])
            loss = next_loss + mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedular.step(epoch)
            avg_loss += loss.item()
        print('Average loss: {}'.format(avg_loss / len(data_iter)))


# if __name__ == '__main__':
#     train()
    # num_processes = 4
    # NOTE: this is required for the ``fork`` method to work
    # model.share_memory()
    # processes = []
    # import torch.multiprocessing as mp
    # for rank in range(num_processes):
    #     p = mp.Process(target=train, args=())
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-P", "--path", dest="path",
                        help="sample training file",
                        default='sample-data/bert-example.txt')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--dim-model", dest="dim_model",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=80, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)