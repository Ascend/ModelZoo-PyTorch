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
Original repository: https://github.com/xiaomengyc/SPG
"""

import torch
import torch.nn as nn

from .util import get_attention
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

__all__ = ['spg']


def compute_attention(feat_map, labels, logits_b1, logits_b2):
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    attention = get_attention(upsample_module(feat_map), labels)
    fused_attention = _get_fused_attention(logits_b1, logits_b2)
    return attention, fused_attention


def _get_fused_attention(feature1, feature2):
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    feat_map1 = feature1.detach().clone()
    feat_map2 = feature2.detach().clone()
    return (torch.sigmoid(upsample_module(feat_map1)) +
            torch.sigmoid(upsample_module(feat_map2))) / 2.


def _get_loss_attention(logits, pre_mask, high_thr, low_thr):
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    mask = get_mask(pre_mask, high_thr, low_thr)
    return loss_attention(loss_func=nn.BCEWithLogitsLoss(),
                          logits=upsample_module(logits).squeeze(dim=1),
                          labels=mask)


def get_loss(output_dict, target, spg_thresholds):
    (h1, l1), (h2, l2), (h3, l3) = spg_thresholds
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    b2i = torch.sigmoid(upsample_module(output_dict['logits_b2']))

    loss_cls = nn.CrossEntropyLoss()(output_dict['logits'], target.long())
    loss_b2_att = _get_loss_attention(
        logits=output_dict['logits_b2'],
        pre_mask=output_dict['attention'],
        high_thr=h1,
        low_thr=l1)
    loss_b1_b2i = _get_loss_attention(
        logits=output_dict['logits_b1'],
        pre_mask=b2i,
        high_thr=h2,
        low_thr=l2)
    loss_c1_fus = _get_loss_attention(
        logits=output_dict['logits_c'],
        pre_mask=output_dict['fused_attention'],
        high_thr=h3,
        low_thr=l3)

    return loss_cls + loss_b2_att + loss_b1_b2i + loss_c1_fus


def mask_fg(mask, attention, threshold):
    for batch_idx in range(attention.size(0)):
        bool_fg = attention[batch_idx] > threshold
        if torch.sum(bool_fg.float()).item() < 30:
            new_threshold = torch.max(attention[batch_idx]) * 0.7
            bool_fg = attention[batch_idx] > new_threshold
        mask[batch_idx][bool_fg] = 1.
    return mask


def mask_bg(mask, attention, threshold=0.05):
    pos_bg = attention < threshold
    mask[pos_bg.data] = 0.
    return mask


def get_mask(attention, thr_high, thr_low):
    mask = attention.new_zeros((attention.size(0), 1, 224, 224)).fill_(255)
    mask = mask_fg(mask, attention, thr_high)
    mask = mask_bg(mask, attention, thr_low)
    return mask


def loss_attention(loss_func, logits, labels):
    pos = labels.view(-1, 1) < 255.
    return loss_func(logits.view(-1, 1)[pos],
                     labels.view(-1, 1)[pos].detach().clone())
