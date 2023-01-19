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
from typing import Tuple

import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['convex_iou', 'convex_giou'])


def convex_giou(pointsets: torch.Tensor,
                polygons: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return generalized intersection-over-union (Jaccard index) between point
    sets and polygons.

    Args:
        pointsets (torch.Tensor): It has shape (N, 18),
            indicating (x1, y1, x2, y2, ..., x9, y9) for each row.
        polygons (torch.Tensor): It has shape (N, 8),
            indicating (x1, y1, x2, y2, x3, y3, x4, y4) for each row.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The first element is the gious
        between point sets and polygons with the shape (N,). The second
        element is the gradient of point sets with the shape (N, 18).
    """
    output = pointsets.new_zeros((pointsets.size(0), 19))
    ext_module.convex_giou(pointsets, polygons, output)
    convex_giou = output[:, -1]
    points_grad = output[:, 0:-1]
    return convex_giou, points_grad


def convex_iou(pointsets: torch.Tensor,
               polygons: torch.Tensor) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) between point sets and
    polygons.

    Args:
        pointsets (torch.Tensor): It has shape (N, 18),
            indicating (x1, y1, x2, y2, ..., x9, y9) for each row.
        polygons (torch.Tensor): It has shape (K, 8),
            indicating (x1, y1, x2, y2, x3, y3, x4, y4) for each row.

    Returns:
        torch.Tensor: Return the ious between point sets and polygons with the
        shape (N, K).
    """
    N, K = pointsets.size(0), polygons.size(0)
    ious = pointsets.new_zeros((N, K))
    ext_module.convex_iou(pointsets, polygons, ious)
    return ious
