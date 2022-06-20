# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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
#
#
import torch
from torch.nn import functional as F

# TODO: merge these two function
def heatmap_focal_loss(
    inputs,
    targets,
    pos_inds,
    labels,
    alpha: float = -1,
    beta: float = 4,
    gamma: float = 2,
    reduction: str = 'sum',
    sigmoid_clamp: float = 1e-4,
    ignore_high_fp: float = -1.,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs:  (sum_l N*Hl*Wl, C)
        targets: (sum_l N*Hl*Wl, C)
        pos_inds: N
        labels: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1-sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred_pix = pred[pos_inds] # N x C
    pos_pred = pos_pred_pix.gather(1, labels.unsqueeze(1))
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights

    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    if reduction == "sum":
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return - pos_loss, - neg_loss

heatmap_focal_loss_jit = torch.jit.script(heatmap_focal_loss)
# heatmap_focal_loss_jit = heatmap_focal_loss

def binary_heatmap_focal_loss(
    inputs,
    targets,
    pos_inds,
    num_pos_avg,
    alpha: float = -1,
    beta: float = 4,
    gamma: float = 2,
    sigmoid_clamp: float = 1e-4,
    ignore_high_fp: float = -1.,
):
    """
    Args:
        inputs:  (sum_l N*Hl*Wl,)
        targets: (sum_l N*Hl*Wl,)
        pos_inds: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1-sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred = pred[pos_inds] # N
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma) / num_pos_avg
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights / num_pos_avg
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = - pos_loss.sum()
    neg_loss = - neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return pos_loss, neg_loss

# binary_heatmap_focal_loss_jit = torch.jit.script(binary_heatmap_focal_loss)
binary_heatmap_focal_loss_jit = binary_heatmap_focal_loss