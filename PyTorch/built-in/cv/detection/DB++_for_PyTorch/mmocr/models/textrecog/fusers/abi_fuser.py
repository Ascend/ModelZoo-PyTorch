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
from mmcv.runner import BaseModule

from mmocr.models.builder import FUSERS


@FUSERS.register_module()
class ABIFuser(BaseModule):
    """Mix and align visual feature and linguistic feature Implementation of
    language model of `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        d_model (int): Hidden size of input.
        max_seq_len (int): Maximum text sequence length :math:`T`.
        num_chars (int): Number of text characters :math:`C`.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 d_model=512,
                 max_seq_len=40,
                 num_chars=90,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.max_seq_len = max_seq_len + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_chars)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and
                d is dim of model.
            v_feature: (N, T, E) shape the same as l_feature.

        Returns:
            A dict with key ``logits``
            The logits of shape (N, T, C) where N is batch size, T is length
                and C is the number of characters.
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)

        return {'logits': logits}
