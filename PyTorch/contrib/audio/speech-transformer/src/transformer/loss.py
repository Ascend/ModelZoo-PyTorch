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
import torch
import torch.nn.functional as F

from utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)
    loss = cal_loss(pred, gold, smoothing)
    return loss

def cal_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)
        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = (loss * non_pad_mask.float()).sum() / n_word
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')
    return loss
