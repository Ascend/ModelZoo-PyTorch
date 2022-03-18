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
Created by raj at 2:31 PM,  2/20/20
"""
from torch.utils.data.dataloader import DataLoader

from dataset.bilingual_data_iter import MyIterableDataset
from dataset.iwslt_data import rebatch_data
from dataset.vocab import WordVocab
from models.utils.model_utils import my_collate
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

src = 'en'
trg = 'it'
input_file = 'sample-data/europarl.enc'
for lang in (src, trg):
    with open(input_file + '.' + lang) as f:
        vocab = WordVocab(f)
        vocab.save_vocab("sample-data/{}.pkl".format(lang))

vocab_src = WordVocab.load_vocab("sample-data/{}.pkl".format(src))
vocab_trg = WordVocab.load_vocab("sample-data/{}.pkl".format(trg))

# Only useful in case we don't need shuffling of data
dataset = MyIterableDataset(filename='sample-data/europarl.enc', src=src, trg=trg,
                         vocab_src=vocab_src, vocab_trg=vocab_trg)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=my_collate)
for epoch in range(10):
    for i, batch in enumerate(rebatch_data(batch=b, pad_idx=1, device='cpu') for b in dataloader):
        print(batch.src, batch.trg)

    print(epoch)
