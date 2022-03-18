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

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MLHingeLoss(nn.Module):

    def forward(self, x, y, reduction='mean'):
        """
            y: labels have standard {0,1} form and will be converted to indices
        """
        b, c = x.size()
        idx = (torch.arange(c) + 1).type_as(x)
        y_idx, _ = (idx * y).sort(-1, descending=True)
        y_idx = (y_idx - 1).long()

        return F.multilabel_margin_loss(x, y_idx, reduction=reduction)

def get_criterion(loss_name, **kwargs):

    losses = {
            "SoftMargin": nn.MultiLabelSoftMarginLoss,
            "Hinge": MLHingeLoss
            }

    return losses[loss_name](**kwargs)


#
# Mask self-supervision
#
def mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    weight = pseudo_gt.sum(1).type_as(mask_gt)
    mask_gt += (1 - weight) * ignore_index

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index)
    return loss.mean()
