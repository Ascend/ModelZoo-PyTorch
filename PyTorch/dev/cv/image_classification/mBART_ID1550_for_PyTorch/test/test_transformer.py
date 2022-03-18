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


import torch
import numpy as np

from models.transformer import SelfAttention, TransformerBlock, TransformerEncoder, TransformerEncoderDecoder

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05 
Date: January 18, 2020	
"""

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader import BertDataSet
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

with open("experiments/sample-data/bert-example.txt") as f:
    vocab = WordVocab(f)
    vocab.save_vocab("experiments/sample-data/vocab.pkl")

vocab = WordVocab.load_vocab("experiments/sample-data/vocab.pkl")

lr_warmup = 500
batch_size = 16
k=512
h=4
depth=1
max_size=80
data_set = BertDataSet("experiments/sample-data/bert-example.txt", vocab, max_size)

data_loader = DataLoader(data_set, batch_size=batch_size)
vocab_size = len(vocab.stoi)
model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size, num_emb_target=vocab_size, max_len=max_size)

criterion = nn.NLLLoss(ignore_index=0)
optimizer = Adam(lr=0.0001, params=model.parameters())
lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

cuda_condition = torch.npu.is_available()
device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

if cuda_condition:
    model.npu()

if cuda_condition and torch.npu.device_count() > 1:
    print("Using %d GPUS for BERT" % torch.npu.device_count())
    model = nn.DataParallel(model, device_ids=[0,1,2,3])


for epoch in range(100):
    avg_loss = 0
    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Running epoch: {}".format(epoch),
                          total=len(data_loader))
    for i, data in data_iter:
        data = {key: value.to(f'npu:{NPU_CALCULATE_DEVICE}') for key, value in data.items()}
        bert_input, bert_label, segment_label, is_next = data
        mask_out, sentence_pred = model(data[bert_input], data[segment_label])

        mask_loss = criterion(mask_out.transpose(1, 2), data[bert_label])
        next_loss = criterion(sentence_pred, data[is_next])
        loss = next_loss + mask_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedular.step()
        avg_loss += loss.item()
    print(avg_loss)