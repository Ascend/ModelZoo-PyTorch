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
from mmcv.runner import Sequential

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import BidirectionalLSTM
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class CRNNDecoder(BaseDecoder):
    """Decoder for CRNN.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        rnn_flag (bool): Use RNN or CNN as the decoder.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 rnn_flag=False,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.rnn_flag = rnn_flag

        if rnn_flag:
            self.decoder = Sequential(
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, num_classes))
        else:
            self.decoder = nn.Conv2d(
                in_channels, num_classes, kernel_size=1, stride=1)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        assert feat.size(2) == 1, 'feature height must be 1'
        if self.rnn_flag:
            x = feat.squeeze(2)  # [N, C, W]
            x = x.permute(2, 0, 1)  # [W, N, C]
            x = self.decoder(x)  # [W, N, C]
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            x = self.decoder(feat)
            x = x.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = x.size()
            outputs = x.view(n, w, c * h)
        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        return self.forward_train(feat, out_enc, None, img_metas)
