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
Created by raj at 15:47 
Date: February 15, 2020	
"""
import math
import os
from argparse import ArgumentParser
from math import inf
import time

import torch
import tqdm
from torch import nn
from torch.autograd import Variable

from dataset.iwslt_data import get_data, MyIterator, batch_size_fn, rebatch, SimpleLossCompute, LabelSmoothing, \
    subsequent_mask, save_dataset

from dataset.iwslt_data import NoamOpt
from models.decoding import greedy_decode, beam_search
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import save_state, load_model_state, get_perplexity
from optim.lr_warm_up import GradualWarmupScheduler
from options import get_parser
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def train(arg):
    model_dir = arg.model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    train, val, test, SRC, TGT = get_data(arg)
    pad_idx = TGT.vocab.stoi["<blank>"]

    BATCH_SIZE = arg.batch_size
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length

    n_batches = math.ceil(len(train) / BATCH_SIZE)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                            device=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                            device=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=arg.dropout,
                                      depth=depth,
                                      num_emb=len(SRC.vocab),
                                      num_emb_target=len(TGT.vocab),
                                      max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    start_epoch = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model,
                                   data_parallel=arg.data_parallel)

    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=arg.label_smoothing)
    optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(),
                                                             lr=arg.lr,
                                                             betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=1000,
    #                                           after_scheduler=scheduler_cosine)

    cuda_condition = torch.npu.is_available() and not arg.cpu
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    if cuda_condition:
        model.npu()

    if cuda_condition and torch.npu.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.npu.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    previous_best = inf

    for epoch in range(start_epoch, arg.num_epochs):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(rebatch(pad_idx, b, device=device) for b in train_iter):
            model.train()
            # bs = batch.batch_size
            # tgt_lengths = (trg != pad_idx).data.sum(dim=1)
            # src_lengths = (src != pad_idx).data.sum(dim=1)
            # batch_ntokens = tgt_lengths.sum()
            # src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
            out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
            loss = compute_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % arg.wait == 0 and i > 0:
                elapsed = time.time() - start
                print("Epoch %d Step: %d Loss: %f PPL: %f Tokens per Sec: %f" %
                      (epoch, i, loss / batch.ntokens, get_perplexity(loss / batch.ntokens), tokens / elapsed))
                start = time.time()
                tokens = 0
                # checkpoint = "checkpoint.{}.".format(total_loss / total_tokens) + 'epoch' + str(epoch) + ".pt"
                # save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)
        loss_average = total_loss / total_tokens
        checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(epoch) + ".pt"
        save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)

        if previous_best > loss_average:
            save_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, criterion, optimizer, epoch)
            previous_best = loss_average


def decode(arg):
    model_dir = arg.model
    train, val, test, SRC, TGT = get_data(arg)
    pad_idx = TGT.vocab.stoi["<blank>"]
    # for decoding keep the batch size one as beam-search is not yet implemented to handle the batch decoding
    BATCH_SIZE = 1
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length

    n_batches = math.ceil(len(train) / BATCH_SIZE)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                            device=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                            device=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'),
                            repeat=False, train=False)

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=arg.dropout, depth=depth,
                                      num_emb=len(SRC.vocab),
                                      num_emb_target=len(TGT.vocab), max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, data_parallel=arg.data_parallel)
    model.eval()

    cuda_condition = torch.npu.is_available() and not arg.cpu
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    if cuda_condition:
        model.npu()

    # Setting the tqdm progress bar
    # data_iter = tqdm.tqdm(enumerate(data_loader),
    #                       desc="Decoding",
    #                       total=len(data_loader))

    translated = list()
    reference = list()

    with torch.no_grad():
        for k, batch in enumerate(rebatch(pad_idx, b, device=device) for b in valid_iter):
            print('Processing: {0}'.format(k))
            start_symbol = TGT.vocab.stoi["<sos>"]
            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=start_symbol)
            out = beam_search(model, batch.src, batch.src_mask, start_symbol=start_symbol, pad_symbol=pad_idx,
                              max=batch.ntokens + 10)

            # print("Source:", end="\t")
            # for i in range(1, batch.src.size(1)):
            #     sym = SRC.vocab.itos[batch.src.data[0, i]]
            #     if sym == "<eos>": break
            #     print(sym, end=" ")
            # print()
            # print("Translation:", end="\t")

            transl = list()
            start_idx = 0 # for greedy decoding the start index should be 1 that will exclude the <sos> symbol
            for i in range(start_idx, out.size(1)):
                sym = TGT.vocab.itos[out[0, i]]
                if sym == "<eos>": break
                transl.append(sym)
            translated.append(' '.join(transl))

            # print()
            # print("Target:", end="\t")
            # ref = list()
            # for i in range(1, batch.trg.size(1)):
            #     sym = TGT.vocab.itos[batch.trg.data[0, i]]
            #     if sym == "<eos>": break
            #     ref.append(sym)
            # reference.append(" ".join(ref))

    with open('valid-beam-decode-test.de-en.en', 'w') as outfile:
        outfile.write('\n'.join(translated))
    # with open('valid-ref.de-en.en', 'w') as outfile:
    #     outfile.write('\n'.join(reference))


def main():
    options = get_parser()
    if options.train:
        print('Launching training...')
        train(options)
    elif options.decode:
        print('Launching decoding...')
        decode(options)
    else:
        print("Specify either --train or --decode")


if __name__ == "__main__":
    main()
