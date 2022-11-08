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
from mmdet.models.detectors import MaskRCNN

from mmocr.core import seg2boundary
from mmocr.models.builder import DETECTORS
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class OCRMaskRCNN(TextDetectorMixin, MaskRCNN):
    """Mask RCNN tailored for OCR."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 text_repr_type='quad',
                 show_score=False,
                 init_cfg=None):
        TextDetectorMixin.__init__(self, show_score)
        MaskRCNN.__init__(
            self,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        assert text_repr_type in ['quad', 'poly']
        self.text_repr_type = text_repr_type

    def get_boundary(self, results):
        """Convert segmentation into text boundaries.

        Args:
           results (tuple): The result tuple. The first element is
               segmentation while the second is its scores.
        Returns:
           dict: A result dict containing 'boundary_result'.
        """

        assert isinstance(results, tuple)

        instance_num = len(results[1][0])
        boundaries = []
        for i in range(instance_num):
            seg = results[1][0][i]
            score = results[0][0][i][-1]
            boundary = seg2boundary(seg, self.text_repr_type, score)
            if boundary is not None:
                boundaries.append(boundary)

        results = dict(boundary_result=boundaries)
        return results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):

        results = super().simple_test(img, img_metas, proposals, rescale)

        boundaries = self.get_boundary(results[0])
        boundaries = boundaries if isinstance(boundaries,
                                              list) else [boundaries]
        return boundaries
