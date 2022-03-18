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

import math
import torch
from .multihead_attention import MultiheadAttention


class SparseMultiheadAttention(MultiheadAttention):
    """ Sparse Multi-Headed Attention.

    "Generating Long Sequences with Sparse Transformers". Implements
    fixed factorized self attention, where l=stride and c=expressivity.
    A(1) includes all words in the stride window and A(2) takes a summary of c
    words from the end of each stride window.
    If is_bidirectional=False, we do not include any words past the current word,
    as in the paper.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, stride=32, expressivity=8, is_bidirectional=True):

        super().__init__(
            embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv,
            add_zero_attn, self_attention, encoder_decoder_attention
        )

        self.is_bidirectional = is_bidirectional
        self.stride = stride
        self.expressivity = expressivity
        assert(self.stride > 0 and self.stride >= self.expressivity)

    # Used for Ai(2) calculations - beginning of [l-c, l] range
    def compute_checkpoint(self, word_index):
        if word_index % self.stride == 0 and word_index != 0:
            checkpoint_index = word_index - self.expressivity
        else:
            checkpoint_index = (
                math.floor(word_index / self.stride) * self.stride
                + self.stride - self.expressivity
            )
        return checkpoint_index

    # Computes Ai(2)
    def compute_subset_summaries(self, absolute_max):
        checkpoint_index = self.compute_checkpoint(0)
        subset_two = set()
        while checkpoint_index <= absolute_max-1:
            summary = set(range(checkpoint_index, min(
                checkpoint_index+self.expressivity+1, absolute_max)
            ))
            subset_two = subset_two.union(summary)
            checkpoint_index = self.compute_checkpoint(checkpoint_index+self.stride)
        return subset_two

    # Sparse Transformer Fixed Attention Pattern: https://arxiv.org/pdf/1904.10509.pdf
    def compute_fixed_attention_subset(self, word_index, tgt_len):
        # +1s account for range function; [min, max) -> [min, max]
        if not self.is_bidirectional:
            absolute_max = word_index + 1
        else:
            absolute_max = tgt_len

        # Subset 1 - whole window
        rounded_index = math.floor((word_index + self.stride) / self.stride) * self.stride
        if word_index % self.stride == 0 and word_index != 0:
            subset_one = set(range(word_index-self.stride, min(absolute_max, word_index+1)))
        else:
            subset_one = set(range(max(0, rounded_index - self.stride), min(
                absolute_max, rounded_index+1))
            )

        # Subset 2 - summary per window
        # If bidirectional, subset 2 is the same for every index
        subset_two = set()
        if not self.is_bidirectional:
            subset_two = self.compute_subset_summaries(absolute_max)

        return subset_one.union(subset_two)

    # Compute sparse mask - if bidirectional, can pre-compute and store
    def buffered_sparse_mask(self, tensor, tgt_len, src_len):
        assert(tgt_len > self.stride)
        sparse_mask = torch.empty((tgt_len, src_len)).float().fill_(float('-10000'))

        # If bidirectional, subset 2 is the same for every index
        subset_summaries = set()
        if self.is_bidirectional:
            subset_summaries = self.compute_subset_summaries(tgt_len)

        for i in range(tgt_len):
            fixed_attention_subset = self.compute_fixed_attention_subset(i, tgt_len)
            fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
            included_word_indices = torch.LongTensor(list(fixed_attention_subset))
            sparse_mask[i].index_fill_(0, included_word_indices, 0)
        return sparse_mask.type_as(tensor)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        sparse_mask = self.buffered_sparse_mask(attn_weights, tgt_len, src_len)
        sparse_mask = sparse_mask.unsqueeze(0).expand(bsz * self.num_heads, tgt_len, src_len)
        attn_weights += sparse_mask
