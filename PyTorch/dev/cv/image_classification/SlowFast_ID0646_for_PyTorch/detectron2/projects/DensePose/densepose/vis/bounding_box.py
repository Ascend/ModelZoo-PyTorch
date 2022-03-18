#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Facebook, Inc. and its affiliates.
from .base import RectangleVisualizer, TextVisualizer


class BoundingBoxVisualizer(object):
    def __init__(self):
        self.rectangle_visualizer = RectangleVisualizer()

    def visualize(self, image_bgr, boxes_xywh):
        for bbox_xywh in boxes_xywh:
            image_bgr = self.rectangle_visualizer.visualize(image_bgr, bbox_xywh)
        return image_bgr


class ScoredBoundingBoxVisualizer(object):
    def __init__(self, bbox_visualizer_params=None, score_visualizer_params=None, **kwargs):
        if bbox_visualizer_params is None:
            bbox_visualizer_params = {}
        if score_visualizer_params is None:
            score_visualizer_params = {}
        self.visualizer_bbox = RectangleVisualizer(**bbox_visualizer_params)
        self.visualizer_score = TextVisualizer(**score_visualizer_params)

    def visualize(self, image_bgr, scored_bboxes):
        boxes_xywh, box_scores = scored_bboxes
        assert len(boxes_xywh) == len(
            box_scores
        ), "Number of bounding boxes {} should be equal to the number of scores {}".format(
            len(boxes_xywh), len(box_scores)
        )
        for i, box_xywh in enumerate(boxes_xywh):
            score_i = box_scores[i]
            image_bgr = self.visualizer_bbox.visualize(image_bgr, box_xywh)
            score_txt = "{0:6.4f}".format(score_i)
            topleft_xy = box_xywh[0], box_xywh[1]
            image_bgr = self.visualizer_score.visualize(image_bgr, score_txt, topleft_xy)
        return image_bgr
