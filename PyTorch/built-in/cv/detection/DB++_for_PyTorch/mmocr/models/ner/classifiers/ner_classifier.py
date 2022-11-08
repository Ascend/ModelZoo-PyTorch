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
from mmocr.models.builder import (DETECTORS, build_convertor, build_decoder,
                                  build_encoder, build_loss)
from mmocr.models.textrecog.recognizer.base import BaseRecognizer


@DETECTORS.register_module()
class NerClassifier(BaseRecognizer):
    """Base class for NER classifier."""

    def __init__(self,
                 encoder,
                 decoder,
                 loss,
                 label_convertor,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.label_convertor = build_convertor(label_convertor)

        self.encoder = build_encoder(encoder)

        decoder.update(num_labels=self.label_convertor.num_labels)
        self.decoder = build_decoder(decoder)

        loss.update(num_labels=self.label_convertor.num_labels)
        self.loss = build_loss(loss)

    def extract_feat(self, imgs):
        """Extract features from images."""
        raise NotImplementedError(
            'Extract feature module is not implemented yet.')

    def forward_train(self, imgs, img_metas, **kwargs):
        encode_out = self.encoder(img_metas)
        logits, _ = self.decoder(encode_out)
        loss = self.loss(logits, img_metas)
        return loss

    def forward_test(self, imgs, img_metas, **kwargs):
        encode_out = self.encoder(img_metas)
        _, preds = self.decoder(encode_out)
        pred_entities = self.label_convertor.convert_pred2entities(
            preds, img_metas['attention_masks'])
        return pred_entities

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('Augmentation test is not implemented yet.')

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('Simple test is not implemented yet.')
