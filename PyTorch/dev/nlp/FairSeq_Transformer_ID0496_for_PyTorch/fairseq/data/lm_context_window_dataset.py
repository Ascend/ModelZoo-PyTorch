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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data.monolingual_dataset import MonolingualDataset

from . import FairseqDataset


class LMContextWindowDataset(FairseqDataset):
    """Wraps a MonolingualDataset and provides more context for evaluation."""

    def __init__(self, dataset, tokens_per_sample, context_window, pad_idx):
        assert isinstance(dataset, MonolingualDataset)
        assert context_window > 0
        self.dataset = dataset
        self.tokens_per_sample = tokens_per_sample
        self.context_window = context_window
        self.pad_idx = pad_idx
        self.prev_tokens = np.empty([0])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        sample = self.dataset.collater(samples)

        pad = self.pad_idx
        max_sample_len = self.tokens_per_sample + self.context_window

        bsz, tsz = sample['net_input']['src_tokens'].shape
        start_idxs = [0] * bsz
        toks = sample['net_input']['src_tokens']
        lengths = sample['net_input']['src_lengths']
        tgt = sample['target']
        new_toks = np.empty([bsz, tsz + self.context_window], dtype=np.int64)
        new_tgt = np.full([bsz, tsz + self.context_window], pad, dtype=np.int64)
        sample_lens = toks.ne(pad).long().sum(dim=1).cpu()
        for i in range(bsz):
            sample_len = sample_lens[i]
            extra = len(self.prev_tokens) + sample_len - max_sample_len
            if extra > 0:
                self.prev_tokens = self.prev_tokens[extra:]
            pads = np.full(self.context_window - len(self.prev_tokens), pad)
            new_toks[i] = np.concatenate([self.prev_tokens, toks[i].numpy(), pads])
            new_tgt[i, len(self.prev_tokens):len(self.prev_tokens) + len(tgt[i])] = tgt[i]
            start_idxs[i] = len(self.prev_tokens)
            lengths[i] += len(self.prev_tokens)
            self.prev_tokens = new_toks[i][new_toks[i] != pad][-self.context_window:]
        sample['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        sample['target'] = torch.from_numpy(new_tgt)
        sample['start_indices'] = start_idxs

        return sample

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        # NOTE we don't shuffle the data to retain access to the previous dataset elements
        return np.arange(len(self.dataset))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
