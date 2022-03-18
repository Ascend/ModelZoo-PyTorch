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

from fairseq.data import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        super().__init__(pad, eos, unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
        cls='<cls>',
        sep='<sep>'
    ):
        super().__init__(pad, eos, unk, mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index
