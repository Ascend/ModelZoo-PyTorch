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

import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'points_in_boxes_part_forward', 'points_in_boxes_cpu_forward',
    'points_in_boxes_all_forward'
])


def points_in_boxes_part(points: Tensor, boxes: Tensor) -> Tensor:
    """Find the box in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate.
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz] in
            LiDAR/DEPTH coordinate, (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M). Default background = -1.
    """
    assert points.shape[0] == boxes.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points),
                                       dtype=torch.int).fill_(-1)

    # If manually put the tensor 'points' or 'boxes' on a device
    # which is not the current device, some temporary variables
    # will be created on the current device in the cuda op,
    # and the output will be incorrect.
    # Therefore, we force the current device to be the same
    # as the device of the tensors if it was not.
    # Please refer to https://github.com/open-mmlab/mmdetection3d/issues/305
    # for the incorrect output before the fix.
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    ext_module.points_in_boxes_part_forward(boxes.contiguous(),
                                            points.contiguous(),
                                            box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points: Tensor, boxes: Tensor) -> Tensor:
    """Find all boxes in which each point is (CPU). The CPU version of
    :meth:`points_in_boxes_all`.

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in
            LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M, T). Default background = 0.
    """
    assert points.shape[0] == boxes.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    point_indices = points.new_zeros((batch_size, num_boxes, num_points),
                                     dtype=torch.int)
    for b in range(batch_size):
        ext_module.points_in_boxes_cpu_forward(boxes[b].float().contiguous(),
                                               points[b].float().contiguous(),
                                               point_indices[b])
    point_indices = point_indices.transpose(1, 2)

    return point_indices


def points_in_boxes_all(points: Tensor, boxes: Tensor) -> Tensor:
    """Find all boxes in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M, T). Default background = 0.
    """
    assert boxes.shape[0] == points.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {boxes.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    box_idxs_of_pts = points.new_zeros((batch_size, num_points, num_boxes),
                                       dtype=torch.int).fill_(0)

    # Same reason as line 25-32
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    ext_module.points_in_boxes_all_forward(boxes.contiguous(),
                                           points.contiguous(),
                                           box_idxs_of_pts)

    return box_idxs_of_pts
