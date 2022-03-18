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

import torch

from . import FairseqDataset


class ConcatSentencesDataset(FairseqDataset):

    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        assert all(len(ds) == len(datasets[0]) for ds in datasets), \
            'datasets must have the same length'

    def __getitem__(self, index):
        return torch.cat([ds[index] for ds in self.datasets])

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return sum(ds.sizes for ds in self.datasets)

    def num_tokens(self, index):
        return sum(ds.num_tokens(index) for ds in self.datasets)

    def size(self, index):
        return sum(ds.size(index) for ds in self.datasets)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(
            getattr(ds, 'supports_prefetch', False) for ds in self.datasets
        )

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
