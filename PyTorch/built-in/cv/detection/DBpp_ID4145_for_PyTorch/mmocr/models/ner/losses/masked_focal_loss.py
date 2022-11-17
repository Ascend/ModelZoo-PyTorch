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
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.models.common.losses.focal_loss import FocalLoss


@LOSSES.register_module()
class MaskedFocalLoss(nn.Module):
    """The implementation of masked focal loss.

    The mask has 1 for real tokens and 0 for padding tokens,
        which only keep active parts of the focal loss
    Args:
        num_labels (int): Number of classes in labels.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, num_labels=None, ignore_index=0):
        super().__init__()
        self.num_labels = num_labels
        self.criterion = FocalLoss(ignore_index=ignore_index)

    def forward(self, logits, img_metas):
        '''Loss forword.
        Args:
            logits: Model output with shape [N, C].
            img_metas (dict): A dict containing the following keys:
                    - img (list]): This parameter is reserved.
                    - labels (list[int]): The labels for each word
                        of the sequence.
                    - texts (list): The words of the sequence.
                    - input_ids (list): The ids for each word of
                        the sequence.
                    - attention_mask (list): The mask for each word
                        of the sequence. The mask has 1 for real tokens
                        and 0 for padding tokens. Only real tokens are
                        attended to.
                    - token_type_ids (list): The tokens for each word
                        of the sequence.
        '''

        labels = img_metas['labels']
        attention_masks = img_metas['attention_masks']

        # Only keep active parts of the loss
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(
                logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss_cls': loss}
