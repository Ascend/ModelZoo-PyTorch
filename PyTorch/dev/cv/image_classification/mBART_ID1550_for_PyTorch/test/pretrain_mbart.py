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
import os
import time
from argparse import ArgumentParser

from dataset.iwslt_data import LabelSmoothing, NoamOpt, SimpleLossCompute, rebatch_mbert
from models.transformer import TransformerEncoderDecoder

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.data_loader_mbart import MBartDataSet
from dataset.vocab import WordVocab
from models.utils.model_utils import save_state, my_collate, get_perplexity
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def go(arg):
    input_file = arg.path
    with open(input_file) as f:
        vocab = WordVocab(f)
        vocab.save_vocab("sample-data/vocab.pkl")

    vocab = WordVocab.load_vocab("../sample-data/vocab.pkl")

    lr_warmup = arg.lr_warmup
    batch_size = arg.batch_size
    k = arg.dim_model
    h = arg.num_heads
    depth = arg.depth
    max_size=arg.max_length
    model_dir = "../bert"
    try:
        os.makedirs(model_dir)
    except OSError:
        pass
    data_set = MBartDataSet(input_file, vocab, max_size)

    data_loader = DataLoader(data_set, batch_size=batch_size,
                             collate_fn=my_collate,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=False,
                             shuffle=True
                             )

    vocab_size = len(vocab.stoi)
    model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size, num_emb_target=vocab_size, max_len=max_size)

    # criterion = nn.NLLLoss(ignore_index=0)
    # optimizer = Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #                  weight_decay=0, amsgrad=False)
    # lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=1, smoothing=0.1)
    optimizer = NoamOpt(arg.dim_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    use_cuda = torch.npu.is_available() and not arg.cpu
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    model.to(f'npu:{NPU_CALCULATE_DEVICE}')

    if use_cuda and torch.npu.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.npu.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    for epoch in range(arg.num_epochs):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(rebatch_mbert(pad_idx=vocab.pad_index, batch=b, device=device) for b in data_loader):
            # For mBert we use both source and target the same: batch.src = batch.trg
            out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
            loss = compute_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % arg.wait == 0 and i > 0:
                if i % arg.wait == 0 and i > 0:
                    elapsed = time.time() - start
                    print("Epoch %d Step: %d Loss: %f PPL: %f Tokens per Sec: %f" %
                          (epoch, i, loss / batch.ntokens, get_perplexity(loss / batch.ntokens), tokens / elapsed))
                    start = time.time()
                    tokens = 0
        loss_average = total_loss / total_tokens
        checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(epoch) + ".pt"
        save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)
        print('Average loss: {}'.format(loss_average))


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
                        default='sample-data/europarl.enc.en')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--cpu", dest="cpu",
                        help="Use CPU computation.).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--dim-model", dest="dim_model",
                        help="Size of the character embeddings.",
                        default=512, type=int)

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
                        default=1000, type=int)

    parser.add_argument("--wait",
                        dest="wait",
                        help="Learning rate warmup.",
                        default=1000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()
    print('OPTIONS ', options)
    go(options)

