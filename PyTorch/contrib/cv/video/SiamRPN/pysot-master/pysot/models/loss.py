# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn.functional as F


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)

    loss = F.nll_loss(pred, label, reduction='none')

    label = label.int()
    pos = (label == 1.).int()
    neg = (label == 0.).int()

    pos_sum = pos.sum()
    neg_sum = neg.sum()

    if pos_sum == 0.:
        loss_pos = (loss * pos).sum()
    else:
        loss_pos = (loss * pos).sum() / pos_sum

    if neg_sum == 0.:
        loss_neg = (loss * neg).sum()
    else:
        loss_neg = (loss * neg).sum() / neg_sum

    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)
