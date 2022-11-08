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
import torch.nn as nn

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 ignore_first_char=False):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)
        self.ignore_first_char = ignore_first_char

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            outputs = outputs[:, :-1, :]

        outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.
            img_metas (None): Unused.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses


@LOSSES.register_module()
class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").

    Warning:
        SARLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self, ignore_index=-1, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class TFLoss(CELoss):
    """Implementation of loss module for transformer.

    Args:
        ignore_index (int, optional): The character index to be ignored in
            loss computation.
        reduction (str): Type of reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").
        flatten (bool): Whether to flatten the vectors for loss computation.

    Warning:
        TFLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets
