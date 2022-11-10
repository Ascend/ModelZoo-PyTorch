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
from mmocr.models.builder import ENCODERS, build_decoder, build_encoder
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ABIVisionModel(BaseEncoder):
    """A wrapper of visual feature encoder and language token decoder that
    converts visual features into text tokens.

    Implementation of VisionEncoder in
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        encoder (dict): Config for image feature encoder.
        decoder (dict): Config for language token decoder.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 encoder=dict(type='TransformerEncoder'),
                 decoder=dict(type='ABIVisionDecoder'),
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Images of shape (N, E, H, W).

        Returns:
            dict: A dict with keys ``feature``, ``logits`` and ``attn_scores``.

            - | feature (Tensor): Shape (N, T, E). Raw visual features for
                language decoder.
            - | logits (Tensor): Shape (N, T, C). The raw logits for
                characters. C is the number of characters.
            - | attn_scores (Tensor): Shape (N, T, H, W). Intermediate result
                for vision-language aligner.
        """
        feat = self.encoder(feat)
        return self.decoder(feat=feat, out_enc=None)
