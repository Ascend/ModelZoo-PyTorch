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
Created by raj at 09:11 
Date: February 20, 2020	
"""
import torch
from torch.utils.data.dataset import IterableDataset
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class MyIterableDataset(IterableDataset):

    def __init__(self, filename, src, trg, vocab_src, vocab_trg):
        # Store the filename in object's memory
        self.filename = filename
        self.src = src
        self.trg = trg
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self, text_src, text_trg):
        src_tokens = self.string_to_index(text_src, vocab=self.vocab_src)
        tgt_tokens = self.string_to_index(text_trg, vocab=self.vocab_trg)
        return src_tokens, tgt_tokens

    def line_mapper(self, line_src, line_trg):
        # Splits the line into text and label and applies preprocessing to the text
        # text, label = line.split(',')
        text_src, text_trg = self.preprocess(line_src, line_trg)
        data = {
            "source": text_src,
            "target": text_trg
        }
        return {key: torch.tensor(value) for key, value in data.items()}

    def __iter__(self):
        # Create an iterator
        file_itr_src = open(self.filename + '.' + self.src)
        file_itr_trg = open(self.filename + '.' + self.trg)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr_src, file_itr_trg)

        return mapped_itr

    def string_to_index(self, sentence, vocab):
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            if tokens[i] == vocab.unk_index:
                print(token, 'is unk')
        return tokens
