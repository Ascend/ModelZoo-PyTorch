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
Created by raj at 20:01 
Date: February 19, 2020	
"""
from torch.utils.data.dataloader import DataLoader

from dataset.data_loader_translation import TranslationDataSet, BySequenceLengthSampler
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
vocab_tgt = WordVocab.load_vocab("sample-data/{}.pkl".format(trg))

data_set = TranslationDataSet(input_file, src, trg, vocab_src, vocab_tgt, 100,
                              add_sos_and_eos=True)

bucket_boundaries = [i*10 for i in range(30)]
batch_sizes = 10

sampler = BySequenceLengthSampler(data_set, bucket_boundaries, batch_sizes)

data_loader = DataLoader(data_set, batch_sampler=sampler, collate_fn=my_collate)
# data_loader = DataLoader(data_set, batch_size=batch_sizes, collate_fn=my_collate)

for i, batch in enumerate(data_loader):
    print(batch)
    if i> 100:
        break
