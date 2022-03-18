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
import torch.nn.functional as F


class MeanPoolGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def forward(self, encoder_out):
        if not (
            hasattr(encoder_out, 'encoder_out')
            and hasattr(encoder_out, 'encoder_padding_mask')
            and encoder_out.encoder_out.size(2) == self.embed_dim
        ):
            raise ValueError('Unexpected format for encoder_out')

        # mean pooling over time
        encoder_padding_mask = encoder_out.encoder_padding_mask  # B x T
        encoder_out = encoder_out.encoder_out.transpose(0, 1)    # B x T x C
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()  # required because of transpose above
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)

        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)
