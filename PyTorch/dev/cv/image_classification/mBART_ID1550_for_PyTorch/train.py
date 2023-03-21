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
Created by raj at 6:59 AM,  7/30/20
"""
import os
import time
from math import inf

from torch import nn

from dataset.iwslt_data import rebatch, rebatch_onmt, SimpleLossCompute, NoamOpt, LabelSmoothing
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import load_model_state, save_state, get_perplexity

"""Train models."""
import torch

if torch.__version__ >= "1.8":
    import torch_npu

import onmt.opts as opts

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, patch_fields, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
import time
import torch.npu
import os
import apex
from apex import amp

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def train(opt):
    if args.ND:
        print('***********allow_internal_format = False*******************')
        torch.npu.config.allow_internal_format = False
    else:
        torch.npu.config.allow_internal_format = True
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=f'npu:{NPU_CALCULATE_DEVICE}')
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    src_vocab = fields['src'].base_field.vocab
    trg_vocab = fields['tgt'].base_field.vocab

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    pad_idx = src_vocab.stoi["<blank>"]
    unk_idx = src_vocab.stoi["<unk>"]
    start_symbol = trg_vocab.stoi["<s>"]

    # patch for fields that may be missing in old data/model
    patch_fields(opt, fields)

    if len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        train_iter = build_dataset_iter(shard_base, fields, opt)

    model_dir = opt.save_model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    model_dim = opt.state_dim
    heads = opt.heads
    depth = opt.enc_layers
    max_len = 100

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=opt.dropout[0],
                                      depth=depth,
                                      num_emb=src_vocab_size,
                                      num_emb_target=trg_vocab_size,
                                      max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    start_steps = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), opt, model,
                                   data_parallel=False)
    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=pad_idx, smoothing=opt.label_smoothing)

    cuda_condition = torch.npu.is_available() and opt.gpu_ranks
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    if cuda_condition:
        model.npu()
    ############# Change Start ##################
    # if cuda_condition and torch.npu.device_count() > 1:
    #    print("Using %d GPUS for BERT" % torch.npu.device_count())
    #    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    ############# Change End ####################

    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(),
                                             lr=opt.learning_rate,
                                             betas=(0.9, 0.98), eps=1e-9)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O2",
                                      combine_grad=True)

    optimizer = NoamOpt(model_dim, 1, 2000, optimizer)
    ############# Change Start ##################
    # optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(),
    #                                                         lr=opt.learning_rate,
    #                                                         betas=(0.9, 0.98), eps=1e-9))
    ############# Change End ####################
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    previous_best = inf
    # start steps defines if training was intrupted
    global_steps = start_steps
    iterations = 0
    max_steps = opt.train_steps
    epochs = opt.epochs
    for epoch in range(epochs):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        iterations += 1
        for i, batch in enumerate(rebatch_onmt(pad_idx, b, device=device) for b in train_iter):
            if opt.max_steps and i > opt.max_steps:
                break
            start_time = time.time()
            global_steps += 1
            model.train()
            out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
            loss = compute_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            step_time = time.time() - start_time
            FPS = opt.batch_size / step_time
            if i < 2:
                print("step_time = {:.4f}".format(step_time), flush=True)
            if i % opt.report_every == 0 and i > 0:
                elapsed = time.time() - start
                print("Epoch: %s, Step: %d Loss: %f PPL: %f Tokens per Sec: %f, time/step(s):%.4f, FPS: %.3f" %
                      (
                      epoch, i, loss / batch.ntokens, get_perplexity(loss / batch.ntokens), tokens / elapsed, step_time,
                      FPS))
                start = time.time()
                tokens = 0
                # checkpoint = "checkpoint.{}.".format(total_loss / total_tokens) + 'epoch' + str(epoch) + ".pt"
                # save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)
        loss_average = total_loss / total_tokens
        checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(iterations) + ".pt"
        save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, global_steps, fields, opt)

        if previous_best > loss_average:
            save_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, criterion, optimizer, global_steps,
                       fields, opt)
            previous_best = loss_average


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    # 开启模糊编译
    if torch.__version__ >= "1.8":
        torch.npu.set_compile_mode(jit_compile=False)
    else:
        torch.npu.global_step_inc()
    parser = _get_parser()
    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
