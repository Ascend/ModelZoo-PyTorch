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

from . import BaseWrapperDataset


class ReplaceDataset(BaseWrapperDataset):
    """Replaces tokens found in the dataset by a specified replacement token

        Args:
            dataset (~torch.utils.data.Dataset): dataset to replace tokens in
            replace_map(Dictionary[int,int]): map of token to replace -> replacement token
            offsets (List[int]): do not replace tokens before (from left if pos, right if neg) this offset. should be
            as many as the number of objects returned by the underlying dataset __getitem__ method.
        """

    def __init__(self, dataset, replace_map, offsets):
        super().__init__(dataset)
        assert len(replace_map) > 0
        self.replace_map = replace_map
        self.offsets = offsets

    def __getitem__(self, index):
        item = self.dataset[index]
        is_tuple = isinstance(item, tuple)
        srcs = item if is_tuple else [item]

        for offset, src in zip(self.offsets, srcs):
            for k, v in self.replace_map.items():
                src_off = src[offset:] if offset >= 0 else src[:offset]
                src_off.masked_fill_(src_off == k, v)

        item = srcs if is_tuple else srcs[0]
        return item
