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
import cv2
import numpy as np
import torch
from mmcv.ops import pixel_group

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor


@POSTPROCESSOR.register_module()
class PANPostprocessor(BasePostprocessor):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_text_area (int): The minimal text instance region area.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_text_confidence=0.5,
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_text_area=16,
                 **kwargs):
        super().__init__(text_repr_type)

        self.min_text_confidence = min_text_confidence
        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_text_area = min_text_area

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        assert preds.dim() == 3

        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        preds = preds.detach().cpu().numpy()

        text_score = preds[0].astype(np.float32)
        text = preds[0] > self.min_text_confidence
        kernel = (preds[1] > self.min_kernel_confidence) * text
        embeddings = preds[2:].transpose((1, 2, 0))  # (h, w, 4)

        region_num, labels = cv2.connectedComponents(
            kernel.astype(np.uint8), connectivity=4)
        contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        kernel_contours = np.zeros(text.shape, dtype='uint8')
        cv2.drawContours(kernel_contours, contours, -1, 255)
        text_points = pixel_group(text_score, text, embeddings, labels,
                                  kernel_contours, region_num,
                                  self.min_text_avg_confidence)

        boundaries = []
        for text_point in text_points:
            text_confidence = text_point[0]
            text_point = text_point[2:]
            text_point = np.array(text_point, dtype=int).reshape(-1, 2)
            area = text_point.shape[0]

            if not self.is_valid_instance(area, text_confidence,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                continue

            vertices_confidence = points2boundary(text_point,
                                                  self.text_repr_type,
                                                  text_confidence)
            if vertices_confidence is not None:
                boundaries.append(vertices_confidence)

        return boundaries
