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


from . import FairseqDataset
from typing import Optional


class TransformEosLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """

    def __init__(
        self,
        dataset: FairseqDataset,
        src_eos: int,
        new_src_eos: Optional[int] = None,
        tgt_bos: Optional[int] = None,
        new_tgt_bos: Optional[int] = None,
    ):
        self.dataset = dataset
        self.src_eos = src_eos
        self.new_src_eos = new_src_eos
        self.tgt_bos = tgt_bos
        self.new_tgt_bos = new_tgt_bos

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        samples = self.dataset.collater(samples)

        # TODO: support different padding direction
        if self.new_src_eos is not None:
            assert(samples['net_input']['src_tokens'][:, -1] != self.src_eos).sum() == 0
            samples['net_input']['src_tokens'][:, -1] = self.new_src_eos

        if self.new_tgt_bos is not None:
            assert (samples['net_input']['prev_output_tokens'][:, 0] != self.tgt_bos).sum() == 0
            samples['net_input']['prev_output_tokens'][:, 0] = self.new_tgt_bos

        return samples

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
