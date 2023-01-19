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

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])


def box_iou_rotated(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    .. note::
        The operator assumes:

        1) The positive direction along x axis is left -> right.

        2) The positive direction along y axis is top -> down.

        3) The w border is in parallel with x axis when angle = 0.

        However, there are 2 opposite definitions of the positive angular
        direction, clockwise (CW) and counter-clockwise (CCW). MMCV supports
        both definitions and uses CW by default.

        Please set ``clockwise=False`` if you are using the CCW definition.

        The coordinate system when ``clockwise`` is ``True`` (default)

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha+0.5h\\sin\\alpha
                \\\\
                y_{center}-0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}


        The coordinate system when ``clockwise`` is ``False``

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (-pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha-0.5h\\sin\\alpha
                \\\\
                y_{center}+0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}

    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious
