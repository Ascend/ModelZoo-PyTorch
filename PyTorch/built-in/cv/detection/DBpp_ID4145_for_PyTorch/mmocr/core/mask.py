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

import mmocr.utils as utils


def points2boundary(points, text_repr_type, text_score=None, min_width=-1):
    """Convert a text mask represented by point coordinates sequence into a
    text boundary.

    Args:
        points (ndarray): Mask index of size (n, 2).
        text_repr_type (str): Text instance encoding type
            ('quad' for quadrangle or 'poly' for polygon).
        text_score (float): Text score.

    Returns:
        boundary (list[float]): The text boundary point coordinates (x, y)
            list. Return None if no text boundary found.
    """
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert text_repr_type in ['quad', 'poly']
    assert text_score is None or 0 <= text_score <= 1

    if text_repr_type == 'quad':
        rect = cv2.minAreaRect(points)
        vertices = cv2.boxPoints(rect)
        boundary = []
        if min(rect[1]) > min_width:
            boundary = [p for p in vertices.flatten().tolist()]

    elif text_repr_type == 'poly':

        height = np.max(points[:, 1]) + 10
        width = np.max(points[:, 0]) + 10

        mask = np.zeros((height, width), np.uint8)
        mask[points[:, 1], points[:, 0]] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boundary = list(contours[0].flatten().tolist())

    if text_score is not None:
        boundary = boundary + [text_score]
    if len(boundary) < 8:
        return None

    return boundary


def seg2boundary(seg, text_repr_type, text_score=None):
    """Convert a segmentation mask to a text boundary.

    Args:
        seg (ndarray): The segmentation mask.
        text_repr_type (str): Text instance encoding type
            ('quad' for quadrangle or 'poly' for polygon).
        text_score (float): The text score.

    Returns:
        boundary (list): The text boundary. Return None if no text found.
    """
    assert isinstance(seg, np.ndarray)
    assert isinstance(text_repr_type, str)
    assert text_score is None or 0 <= text_score <= 1

    points = np.where(seg)
    # x, y order
    points = np.concatenate([points[1], points[0]]).reshape(2, -1).transpose()
    boundary = None
    if len(points) != 0:
        boundary = points2boundary(points, text_repr_type, text_score)

    return boundary


def extract_boundary(result):
    """Extract boundaries and their scores from result.

    Args:
        result (dict): The detection result with the key 'boundary_result'
            of one image.

    Returns:
        boundaries_with_scores (list[list[float]]): The boundary and score
            list.
        boundaries (list[list[float]]): The boundary list.
        scores (list[float]): The boundary score list.
    """
    assert isinstance(result, dict)
    assert 'boundary_result' in result.keys()

    boundaries_with_scores = result['boundary_result']
    assert utils.is_2dlist(boundaries_with_scores)

    boundaries = [b[:-1] for b in boundaries_with_scores]
    scores = [b[-1] for b in boundaries_with_scores]

    return (boundaries_with_scores, boundaries, scores)
