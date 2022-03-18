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

import torch
from itertools import accumulate
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def make_batch_align_matrix(index_tensor, size=None, normalize=False):
    """
    Convert a sparse index_tensor into a batch of alignment matrix,
    with row normalize to the sum of 1 if set normalize.

    Args:
        index_tensor (LongTensor): ``(N, 3)`` of [batch_id, tgt_id, src_id]
        size (List[int]): Size of the sparse tensor.
        normalize (bool): if normalize the 2nd dim of resulting tensor.
    """
    n_fill, device = index_tensor.size(0), index_tensor.device
    value_tensor = torch.ones([n_fill], dtype=torch.float)
    dense_tensor = torch.sparse_coo_tensor(
        index_tensor.t(), value_tensor, size=size, device=device).to_dense()
    if normalize:
        row_sum = dense_tensor.sum(-1, keepdim=True)  # sum by row(tgt)
        # threshold on 1 to avoid div by 0
        torch.nn.functional.threshold(row_sum, 1, 1, inplace=True)
        dense_tensor.div_(row_sum)
    return dense_tensor


def extract_alignment(align_matrix, tgt_mask, src_lens, n_best):
    """
    Extract a batched align_matrix into its src indice alignment lists,
    with tgt_mask to filter out invalid tgt position as EOS/PAD.
    BOS already excluded from tgt_mask in order to match prediction.

    Args:
        align_matrix (Tensor): ``(B, tgt_len, src_len)``,
            attention head normalized by Softmax(dim=-1)
        tgt_mask (BoolTensor): ``(B, tgt_len)``, True for EOS, PAD.
        src_lens (LongTensor): ``(B,)``, containing valid src length
        n_best (int): a value indicating number of parallel translation.
        * B: denote flattened batch as B = batch_size * n_best.

    Returns:
        alignments (List[List[FloatTensor|None]]): ``(batch_size, n_best,)``,
         containing valid alignment matrix (or None if blank prediction)
         for each translation.
    """
    batch_size_n_best = align_matrix.size(0)
    assert batch_size_n_best % n_best == 0

    alignments = [[] for _ in range(batch_size_n_best // n_best)]

    # treat alignment matrix one by one as each have different lengths
    for i, (am_b, tgt_mask_b, src_len) in enumerate(
            zip(align_matrix, tgt_mask, src_lens)):
        valid_tgt = ~tgt_mask_b
        valid_tgt_len = valid_tgt.sum()
        if valid_tgt_len == 0:
            # No alignment if not exist valid tgt token
            valid_alignment = None
        else:
            # get valid alignment (sub-matrix from full paded aligment matrix)
            am_valid_tgt = am_b.masked_select(valid_tgt.unsqueeze(-1)) \
                               .view(valid_tgt_len, -1)
            valid_alignment = am_valid_tgt[:, :src_len]  # only keep valid src
        alignments[i // n_best].append(valid_alignment)

    return alignments


def build_align_pharaoh(valid_alignment):
    """Convert valid alignment matrix to i-j (from 0) Pharaoh format pairs,
    or empty list if it's None.
    """
    align_pairs = []
    if isinstance(valid_alignment, torch.Tensor):
        tgt_align_src_id = valid_alignment.argmax(dim=-1)

        for tgt_id, src_id in enumerate(tgt_align_src_id.tolist()):
            align_pairs.append(str(src_id) + "-" + str(tgt_id))
        align_pairs.sort(key=lambda x: int(x.split('-')[-1]))  # sort by tgt_id
        align_pairs.sort(key=lambda x: int(x.split('-')[0]))  # sort by src_id
    return align_pairs


def to_word_align(src, tgt, subword_align, m_src='joiner', m_tgt='joiner'):
    """Convert subword alignment to word alignment.

    Args:
        src (string): tokenized sentence in source language.
        tgt (string): tokenized sentence in target language.
        subword_align (string): align_pharaoh correspond to src-tgt.
        m_src (string): tokenization mode used in src,
            can be ["joiner", "spacer"].
        m_tgt (string): tokenization mode used in tgt,
            can be ["joiner", "spacer"].

    Returns:
        word_align (string): converted alignments correspand to
            detokenized src-tgt.
    """
    assert m_src in ["joiner", "spacer"], "Invalid value for argument m_src!"
    assert m_tgt in ["joiner", "spacer"], "Invalid value for argument m_tgt!"

    src, tgt = src.strip().split(), tgt.strip().split()
    subword_align = {(int(a), int(b)) for a, b in (x.split("-")
                     for x in subword_align.split())}

    src_map = (subword_map_by_spacer(src, marker='鈻') if m_src == 'spacer'
               else subword_map_by_joiner(src, marker='锟'))

    tgt_map = (subword_map_by_spacer(src, marker='鈻') if m_tgt == 'spacer'
               else subword_map_by_joiner(src, marker='锟'))

    word_align = list({"{}-{}".format(src_map[a], tgt_map[b])
                       for a, b in subword_align})
    word_align.sort(key=lambda x: int(x.split('-')[-1]))  # sort by tgt_id
    word_align.sort(key=lambda x: int(x.split('-')[0]))  # sort by src_id
    return " ".join(word_align)


def subword_map_by_joiner(subwords, marker='锟'):
    """Return word id for each subword token (annotate by joiner)."""
    flags = [0] * len(subwords)
    for i, tok in enumerate(subwords):
        if tok.endswith(marker):
            flags[i] = 1
        if tok.startswith(marker):
            assert i >= 1 and flags[i-1] != 1, \
                "Sentence `{}` not correct!".format(" ".join(subwords))
            flags[i-1] = 1
    marker_acc = list(accumulate([0] + flags[:-1]))
    word_group = [(i - maker_sofar) for i, maker_sofar
                  in enumerate(marker_acc)]
    return word_group


def subword_map_by_spacer(subwords, marker='鈻'):
    """Return word id for each subword token (annotate by spacer)."""
    word_group = list(accumulate([int(marker in x) for x in subwords]))
    if word_group[0] == 1:  # when dummy prefix is set
        word_group = [item - 1 for item in word_group]
    return word_group
