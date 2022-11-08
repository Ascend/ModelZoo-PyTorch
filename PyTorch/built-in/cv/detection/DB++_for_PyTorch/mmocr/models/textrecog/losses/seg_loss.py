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
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class SegLoss(nn.Module):
    """Implementation of loss module for segmentation based text recognition
    method.

    Args:
        seg_downsample_ratio (float): Downsample ratio of
            segmentation map.
        seg_with_loss_weight (bool): If True, set weight for
            segmentation loss.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self,
                 seg_downsample_ratio=0.5,
                 seg_with_loss_weight=True,
                 ignore_index=255,
                 **kwargs):
        super().__init__()

        assert isinstance(seg_downsample_ratio, (int, float))
        assert 0 < seg_downsample_ratio <= 1
        assert isinstance(ignore_index, int)

        self.seg_downsample_ratio = seg_downsample_ratio
        self.seg_with_loss_weight = seg_with_loss_weight
        self.ignore_index = ignore_index

    def seg_loss(self, out_head, gt_kernels):
        seg_map = out_head  # bsz * num_classes * H/2 * W/2
        seg_target = [
            item[1].rescale(self.seg_downsample_ratio).to_tensor(
                torch.long, seg_map.device) for item in gt_kernels
        ]
        seg_target = torch.stack(seg_target).squeeze(1)

        loss_weight = None
        if self.seg_with_loss_weight:
            N = torch.sum(seg_target != self.ignore_index)
            N_neg = torch.sum(seg_target == 0)
            weight_val = 1.0 * N_neg / (N - N_neg)
            loss_weight = torch.ones(seg_map.size(1), device=seg_map.device)
            loss_weight[1:] = weight_val

        loss_seg = F.cross_entropy(
            seg_map,
            seg_target,
            weight=loss_weight,
            ignore_index=self.ignore_index)

        return loss_seg

    def forward(self, out_neck, out_head, gt_kernels):
        """
        Args:
            out_neck (None): Unused.
            out_head (Tensor): The output from head whose shape
                is :math:`(N, C, H, W)`.
            gt_kernels (BitmapMasks): The ground truth masks.

        Returns:
            dict: A loss dictionary with the key ``loss_seg``.
        """

        losses = {}

        loss_seg = self.seg_loss(out_head, gt_kernels)

        losses['loss_seg'] = loss_seg

        return losses
