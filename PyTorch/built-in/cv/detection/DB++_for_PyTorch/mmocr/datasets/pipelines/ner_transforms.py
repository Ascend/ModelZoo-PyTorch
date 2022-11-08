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
from mmdet.datasets.builder import PIPELINES

from mmocr.models.builder import build_convertor


@PIPELINES.register_module()
class NerTransform:
    """Convert text to ID and entity in ground truth to label ID. The masks and
    tokens are generated at the same time. The four parameters will be used as
    input to the model.

    Args:
        label_convertor: Convert text to ID and entity
        in ground truth to label ID.
        max_len (int): Limited maximum input length.
    """

    def __init__(self, label_convertor, max_len):
        self.label_convertor = build_convertor(label_convertor)
        self.max_len = max_len

    def __call__(self, results):
        texts = results['text']
        input_ids = self.label_convertor.convert_text2id(texts)
        labels = self.label_convertor.convert_entity2label(
            results['label'], len(texts))

        attention_mask = [0] * self.max_len
        token_type_ids = [0] * self.max_len
        # The beginning and end IDs are added to the ID,
        # so the mask length is increased by 2
        for i in range(len(texts) + 2):
            attention_mask[i] = 1
        results = dict(
            labels=labels,
            texts=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        return results


@PIPELINES.register_module()
class ToTensorNER:
    """Convert data with ``list`` type to tensor."""

    def __call__(self, results):

        input_ids = torch.tensor(results['input_ids'])
        labels = torch.tensor(results['labels'])
        attention_masks = torch.tensor(results['attention_mask'])
        token_type_ids = torch.tensor(results['token_type_ids'])

        results = dict(
            img=[],
            img_metas=dict(
                input_ids=input_ids,
                attention_masks=attention_masks,
                labels=labels,
                token_type_ids=token_type_ids))
        return results
