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
import copy

from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, ModuleList

from mmocr.models.builder import ENCODERS
from mmocr.models.common.modules import PositionalEncoding


@ENCODERS.register_module()
class TransformerEncoder(BaseModule):
    """Implement transformer encoder for text recognition, modified from
    `<https://github.com/FangShancheng/ABINet>`.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        max_len (int): Maximum output sequence length :math:`T`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=2,
                 n_head=8,
                 d_model=512,
                 d_inner=2048,
                 dropout=0.1,
                 max_len=8 * 32,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.pos_encoder = PositionalEncoding(d_model, n_position=max_len)
        encoder_layer = BaseTransformerLayer(
            operation_order=('self_attn', 'norm', 'ffn', 'norm'),
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=dropout,
            ),
            norm_cfg=dict(type='LN'),
        )
        self.transformer = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, feature):
        """
        Args:
            feature (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.

        Returns:
            Tensor: Features of shape :math:`(N, D_m, H, W)`.
        """
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).transpose(1, 2)  # (n, h*w, c)
        feature = self.pos_encoder(feature)  # (n, h*w, c)
        feature = feature.transpose(0, 1)  # (h*w, n, c)
        for m in self.transformer:
            feature = m(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature
