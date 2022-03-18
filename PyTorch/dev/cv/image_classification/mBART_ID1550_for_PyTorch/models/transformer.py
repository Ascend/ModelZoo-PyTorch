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
Created by raj at 11:05
Date: January 18, 2020
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.embeddings.position_emb import PositionalEncoding
from models.embeddings.token_emb import Embeddings
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class SelfAttention(nn.Module):
    """
    This is basic transformer model
    """

    def __init__(self, emb_dim, heads, mask_future_steps=False, multihead_shared_emb=False):
        super().__init__()
        self.emb_dim, self.heads, self.mask_future_steps = emb_dim, heads, mask_future_steps
        if multihead_shared_emb:
            self.dim_per_head = self.emb_dim // self.heads
        else:
            self.dim_per_head = self.emb_dim

        self.toqueries = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.tovalue = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.tokey = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.unifyheads = nn.Linear(self.dim_per_head * heads, self.emb_dim)

    def forward(self, tensor, mask=None, kv=None):
        bs, qlen, dim = tensor.size()
        if kv is not None:
            kv = kv
        else:
            kv = tensor

        heads = self.heads
        kv_bs, kv_qlen, kv_dim = kv.size()

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        query = self.toqueries(tensor).view(bs, qlen, heads, self.dim_per_head).transpose(1, 2)
        key = self.tokey(kv).view(kv_bs, kv_qlen, heads, self.dim_per_head).transpose(1, 2)
        value = self.tovalue(kv).view(kv_bs, kv_qlen, heads, self.dim_per_head).transpose(1, 2)

        query = query / (self.dim_per_head ** (1 / 4))
        key = key / (self.dim_per_head ** (1 / 4))

        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        dot = F.softmax(scores, dim=-1)
        out = torch.matmul(dot, value)
        out = out.transpose(1, 2).contiguous().view(bs, qlen, heads * self.dim_per_head)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, ff=4, dropout=0.1, multihead_shared_emb=False):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads=heads, multihead_shared_emb=multihead_shared_emb)

        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff * emb_dim, emb_dim))

        self.do = nn.Dropout(dropout)

    def forward(self, tensor, mask=None):
        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.attention(self.norm1(tensor), mask))

        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.ff(self.norm2(tensor)))

        return tensor


class TransformerBlockDecoder(nn.Module):
    def __init__(self, emb_dim, heads, ff=4, mask_future_steps=False, dropout=0.1, multihead_shared_emb=False):
        super().__init__()

        # Masked self attention
        self.attention = SelfAttention(emb_dim, heads=heads, mask_future_steps=mask_future_steps,
                                       multihead_shared_emb=multihead_shared_emb)

        # Encoder-decoder self attention
        self.attention_encoder_decoder = SelfAttention(emb_dim, heads=heads,
                                                       multihead_shared_emb=multihead_shared_emb)

        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(emb_dim, eps=1e-6)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff * emb_dim, emb_dim))

        self.do = nn.Dropout(dropout)

    def forward(self, tensor, memory, src_mask, trg_mask):
        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.attention(self.norm1(tensor), trg_mask))

        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.attention_encoder_decoder(self.norm2(tensor), src_mask, memory))

        # Run feed-forward: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.ff(self.norm3(tensor)))
        return tensor


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, num_tokens, num_classes, dropout=0.1, multihead_shared_emb=True):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = Embeddings(k, num_tokens)
        self.pos_emb = PositionalEncoding(k, dropout)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb_dim=k, heads=heads, multihead_shared_emb=multihead_shared_emb))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        x = self.token_emb(x)
        b, t, k = x.size()

        # generate position embeddings
        x = self.pos_emb(x)

        x = self.do(x)

        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class probabilities
        x = self.toprobs(x.mean(dim=1))

        return F.log_softmax(x, dim=1)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, heads, depth, num_emb, max_len, dropout=0.1, multihead_shared_emb=False):
        super().__init__()
        self.max_len = max_len
        self.token_emb = Embeddings(emb_dim, num_emb)
        self.pos_emb = PositionalEncoding(emb_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(emb_dim, heads, dropout=dropout, multihead_shared_emb=multihead_shared_emb))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, tokens, mask):
        tensor = self.token_emb(tokens)
        tensor = self.pos_emb(tensor)

        for i, layer in enumerate(self.tblocks):
            tensor = layer(tensor, mask)
        return self.norm(tensor)


class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, heads, depth, num_emb_target, max_len, mask_future_steps=False,
                 dropout=0.1, multihead_shared_emb=False):
        super().__init__()
        self.token_emb = Embeddings(emb_dim, num_emb_target)
        self.pos_emb = PositionalEncoding(emb_dim, dropout)
        self.max_len = max_len
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        self.tblocks_decoder = nn.ModuleList()
        for _ in range(depth):
            self.tblocks_decoder.append(TransformerBlockDecoder(emb_dim, heads, mask_future_steps, dropout=dropout,
                                                                multihead_shared_emb=multihead_shared_emb))

    def forward(self, tokens, memory , src_mask, trg_mask):
        tensor = self.token_emb(tokens)
        tensor = self.pos_emb(tensor)
        for i, layer in enumerate(self.tblocks_decoder):
            tensor = layer(tensor, memory, src_mask, trg_mask)
        return self.norm(tensor)


class Generator(nn.Module):
    def __init__(self, k, num_emb_target):
        super().__init__()
        self.ff = nn.Linear(k, num_emb_target)

    def forward(self, enc_dec):
        ff_out = self.ff(enc_dec)
        return F.log_softmax(ff_out, dim=-1), ff_out


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, k, heads, depth, num_emb, num_emb_target, max_len, mask_future_steps=True, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(k, heads, depth, num_emb, max_len, dropout=dropout, multihead_shared_emb=True)
        self.decoder = TransformerDecoder(k, heads, depth, num_emb_target, max_len, mask_future_steps,
                                          dropout=dropout, multihead_shared_emb=True)
        self.generator = Generator(k, num_emb_target)

    def forward(self, src_tokens, src_mask, tgt_tokens=None, trg_mask=None, predict=False):
        memory = self.encoder(src_tokens, src_mask)

        # This code block is to handle mBERT pre-training
        if tgt_tokens is not None:
            tgt_tokens = tgt_tokens
            trg_mask = trg_mask
        else:
            tgt_tokens = src_tokens
            trg_mask = src_mask

        return self.decoder(tgt_tokens, memory, src_mask, trg_mask)
