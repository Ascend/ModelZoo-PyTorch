# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.models.common.losses.dice_loss import DiceLoss


@LOSSES.register_module()
class DBLoss(nn.Module):
    """The class for implementing DBNet loss.

    This is partially adapted from https://github.com/MhLiao/DB.

    Args:
        alpha (float): The binary loss coef.
        beta (float): The threshold loss coef.
        reduction (str): The way to reduce the loss.
        negative_ratio (float): The ratio of positives to negatives.
        eps (float): Epsilon in the threshold loss function.
        bbce_loss (bool): Whether to use balanced bce for probability loss.
            If False, dice loss will be used instead.
    """

    def __init__(self,
                 alpha=1,
                 beta=1,
                 reduction='mean',
                 negative_ratio=3.0,
                 eps=1e-6,
                 bbce_loss=False):
        super().__init__()
        assert reduction in ['mean',
                             'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bbce_loss = bbce_loss
        self.dice_loss = DiceLoss(eps=eps)

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert isinstance(bitmasks, list)
        assert isinstance(target_sz, tuple)

        batch_size = len(bitmasks)
        num_levels = len(bitmasks[0])

        result_tensors = []

        for level_inx in range(num_levels):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx]).npu(non_blocking=True)
                mask_sz = mask.shape
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                if pad[1] > 0 or pad[3] > 0:
                    mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            result_tensors.append(kernel)

        return result_tensors

    def balance_bce_loss(self, pred, gt, mask):

        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))

        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.sort(negative_loss.view(-1), descending=True)
        valid_mask = torch.zeros((len(negative_loss), ))
        valid_mask[:negative_count] = 1
        valid_mask = valid_mask.npu(non_blocking=True)
        negative_loss_sum = (negative_loss * valid_mask).sum()

        balance_loss = (positive_loss.sum() + negative_loss_sum) / (
            positive_count + negative_count + self.eps)

        return balance_loss

    def l1_thr_loss(self, pred, gt, mask):
        thr_loss = torch.abs((pred - gt) * mask).sum() / (
            mask.sum() + self.eps)
        return thr_loss

    def forward(self, preds, downsample_ratio, gt_shrink, gt_shrink_mask,
                gt_thr, gt_thr_mask):
        """Compute DBNet loss.

        Args:
            preds (Tensor): The output tensor with size :math:`(N, 3, H, W)`.
            downsample_ratio (float): The downsample ratio for the
                ground truths.
            gt_shrink (list[BitmapMasks]): The mask list with each element
                being the shrunk text mask for one img.
            gt_shrink_mask (list[BitmapMasks]): The effective mask list with
                each element being the shrunk effective mask for one img.
            gt_thr (list[BitmapMasks]): The mask list with each element
                being the threshold text mask for one img.
            gt_thr_mask (list[BitmapMasks]): The effective mask list with
                each element being the threshold effective mask for one img.

        Returns:
            dict: The dict for dbnet losses with "loss_prob", "loss_db" and
            "loss_thresh".
        """
        assert isinstance(downsample_ratio, float)

        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)
        assert isinstance(gt_thr, list)
        assert isinstance(gt_thr_mask, list)

        pred_prob = preds[:, 0, :, :]
        pred_thr = preds[:, 1, :, :]
        pred_db = preds[:, 2, :, :]
        feature_sz = preds.size()

        keys = ['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            if downsample_ratio != 1:
                gt[k] = [item.rescale(downsample_ratio) for item in gt[k]]
            gt[k] = self.bitmasks2tensor(gt[k], feature_sz[2:])
        gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()
        if self.bbce_loss:
            loss_prob = self.balance_bce_loss(pred_prob, gt['gt_shrink'][0],
                                              gt['gt_shrink_mask'][0])
        else:
            loss_prob = self.dice_loss(pred_prob, gt['gt_shrink'][0],
                                       gt['gt_shrink_mask'][0])

        loss_db = self.dice_loss(pred_db, gt['gt_shrink'][0],
                                 gt['gt_shrink_mask'][0])
        loss_thr = self.l1_thr_loss(pred_thr, gt['gt_thr'][0],
                                    gt['gt_thr_mask'][0])

        results = dict(
            loss_prob=self.alpha * loss_prob,
            loss_db=loss_db,
            loss_thr=self.beta * loss_thr)

        return results
