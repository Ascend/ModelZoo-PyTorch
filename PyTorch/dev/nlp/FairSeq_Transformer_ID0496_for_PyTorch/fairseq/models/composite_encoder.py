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

from fairseq.models import FairseqEncoder


class CompositeEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return a dictionary of outputs. The first
    encoder's dictionary is used for initialization.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        super().__init__(next(iter(encoders.values())).dictionary)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                the outputs from each Encoder
        """
        encoder_out = {}
        for key in self.encoders:
            encoder_out[key] = self.encoders[key](src_tokens, src_lengths)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order."""
        for key in self.encoders:
            encoder_out[key] = self.encoders[key].reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out

    def max_positions(self):
        return min([self.encoders[key].max_positions() for key in self.encoders])

    def upgrade_state_dict(self, state_dict):
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict
