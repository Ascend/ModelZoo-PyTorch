# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) OpenMMLab. All rights reserved.
import math
import operator
from functools import reduce

import numpy as np

from mmocr.utils.check_argument import is_2dlist, is_type_list


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join(
                [x_sorted_boxes[idx]['text'] for idx in box_group])
            x_min, y_min = float('1000'), float('1000')
            x_max, y_max = float('-1000'), float('-1000')
            for idx in box_group:
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
            merged_box['box'] = [
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
            ]
            merged_boxes.append(merged_box)

    return merged_boxes


def bezier_to_polygon(bezier_points, num_sample=20):
    """Sample points from the boundary of a polygon enclosed by two Bezier
    curves, which are controlled by ``bezier_points``.

    Args:
        bezier_points (ndarray): A :math:`(2, 4, 2)` array of 8 Bezeir points
            or its equalivance. The first 4 points control the curve at one
            side and the last four control the other side.
        num_sample (int): The number of sample points at each Bezeir curve.

    Returns:
        list[ndarray]: A list of 2*num_sample points representing the polygon
        extracted from Bezier curves.

    Warning:
        The points are not guaranteed to be ordered. Please use
        :func:`mmocr.utils.sort_points` to sort points if necessary.
    """
    assert num_sample > 0

    bezier_points = np.asarray(bezier_points)
    assert np.prod(
        bezier_points.shape) == 16, 'Need 8 Bezier control points to continue!'

    bezier = bezier_points.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    u = np.linspace(0, 1, num_sample)

    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])

    # Convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.tolist()


def sort_points(points):
    """Sort arbitrary points in clockwise order in Cartesian coordinate, you
    may need to reverse the output sequence if you are using OpenCV's image
    coordinate.

    Reference:
    https://github.com/novioleo/Savior/blob/master/Utils/GeometryUtils.py.

    Warning: This function can only sort convex polygons.

    Args:
        points (list[ndarray] or ndarray or list[list]): A list of unsorted
            boundary points.

    Returns:
        list[ndarray]: A list of points sorted in clockwise order.
    """
    assert is_type_list(points, np.ndarray) or isinstance(points, np.ndarray) \
        or is_2dlist(points)
    center_point = tuple(
        map(operator.truediv,
            reduce(lambda x, y: map(operator.add, x, y), points),
            [len(points)] * 2))
    return sorted(
        points,
        key=lambda coord: (180 + math.degrees(
            math.atan2(*tuple(map(operator.sub, coord, center_point))))) % 360)
