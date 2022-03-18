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
Created by raj at 3:53 PM,  1/15/20
"""
import pickle
from collections import Counter
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


class TorchVocab(object):
    def __init__(self, counter, specials=None, max_size=None, min_freq=1):
        self.freqs = counter
        self.counter = counter.copy()
        self.min_freq = max(min_freq, 1)
        self.max_size = max_size
        self.itos = list(specials)

        for tok in self.itos:
            del counter[tok]
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        def __len__(self):
            return len(self.itos)


class Vocab(TorchVocab):
    def __init__(self, counter, max_size, min_freq):
        super().__init__(counter, specials=['<oov>', '<pad>', '<sos>','<eos>', '<mask>'],
                         max_size=max_size, min_freq=min_freq)
        self.unk_index = 0
        self.pad_index = 1
        self.sos_index = 2
        self.eos_index = 3
        self.mask_index = 4

    def to_seq(self, sentence, max_len=None, with_eos=False, with_sos=False, with_len=False):
        pass

    def from_seq(self, sentence, join=False, with_pad=False):
        pass


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        counter = Counter()
        for text in texts:
            if isinstance(text, list):
                words = text
            else:
                words = text.strip().split()

            for word in words:
                counter[word] += 1

        super().__init__(counter=counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, max_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, list):
            words = sentence
        else:
            words = sentence.split()
        seq = [self.stoi.get(word, self.unk_index) for word in words]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if max_len is None:
            pass
        elif len(seq) <= max_len:
            seq += [self.pad_index for _ in range(max_len - len(seq))]
        else:
            seq = seq[:max_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)