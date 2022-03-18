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
Original repository: https://github.com/xiaomengyc/ACoL
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

__all__ = ['AcolBase']


class AcolBase(nn.Module):
    def _acol_logits(self, feature, labels, drop_threshold):
        feat_map_a, logits = self._branch(feature=feature,
                                          classifier=self.classifier_A)
        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention = get_attention(feature=feat_map_a, label=labels)
        erased_feature = _erase_attention(
            feature=feature, attention=attention, drop_threshold=drop_threshold)
        feat_map_b, logit_b = self._branch(feature=erased_feature,
                                           classifier=self.classifier_B)
        return {'logits': logits, 'logit_b': logit_b,
                'feat_map_a': feat_map_a, 'feat_map_b': feat_map_b}

    def _branch(self, feature, classifier):
        feat_map = classifier(feature)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.size(0), -1)
        return feat_map, logits


def _erase_attention(feature, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = feature * mask
    return erased_feature


def get_loss(output_dict, gt_labels, **kwargs):
    return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
           nn.CrossEntropyLoss()(output_dict['logit_b'], gt_labels.long())
