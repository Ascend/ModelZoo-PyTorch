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
from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['chamfer_distance_forward', 'chamfer_distance_backward'])


class ChamferDistanceFunction(Function):
    """This is an implementation of the 2D Chamfer Distance.

    It has been used in the paper `Oriented RepPoints for Aerial Object
    Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
    """

    @staticmethod
    def forward(ctx, xyz1: Tensor, xyz2: Tensor) -> Sequence[Tensor]:
        """
        Args:
            xyz1 (Tensor): Point set with shape (B, N, 2).
            xyz2 (Tensor): Point set with shape (B, N, 2).

        Returns:
            Sequence[Tensor]:

                - dist1 (Tensor): Chamfer distance (xyz1 to xyz2) with
                    shape (B, N).
                - dist2 (Tensor): Chamfer distance (xyz2 to xyz1) with
                    shape (B, N).
                - idx1 (Tensor): Index of chamfer distance (xyz1 to xyz2)
                    with shape (B, N), which be used in compute gradient.
                - idx2 (Tensor): Index of chamfer distance (xyz2 to xyz2)
                    with shape (B, N), which be used in compute gradient.
        """
        batch_size, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        dist1 = torch.zeros(batch_size, n).to(device)
        dist2 = torch.zeros(batch_size, m).to(device)
        idx1 = torch.zeros(batch_size, n).type(torch.IntTensor).to(device)
        idx2 = torch.zeros(batch_size, m).type(torch.IntTensor).to(device)

        ext_module.chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1,
                                            idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dist1: Tensor, grad_dist2: Tensor,
                 grad_idx1: Tensor,
                 grad_idx2: Tensor) -> Tuple[Tensor, Tensor]:
        """

        Args:
            grad_dist1 (Tensor): Gradient of chamfer distance
                (xyz1 to xyz2) with shape (B, N).
            grad_dist2 (Tensor): Gradient of chamfer distance
                (xyz2 to xyz1) with shape (B, N).
            grad_idx1 (Tensor): Index of chamfer distance (xyz1 to xyz2)
                with shape (B, N), which be used in compute gradient.
            grad_idx2 (Tensor): Index of chamfer distance (xyz2 to xyz2)
                with shape (B, N), which be used in compute gradient.

        Returns:
            Tuple[Tensor, Tensor]:

            - grad_xyz1 (Tensor): Gradient of the point set with shape \
                (B, N, 2).
            - grad_xyz2 (Tensor):Gradient of the point set with shape \
                (B, N, 2).
        """
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        device = grad_dist1.device
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        grad_xyz1 = torch.zeros(xyz1.size()).to(device)
        grad_xyz2 = torch.zeros(xyz2.size()).to(device)

        ext_module.chamfer_distance_backward(xyz1, xyz2, grad_xyz1, grad_xyz2,
                                             grad_dist1, grad_dist2, idx1,
                                             idx2)
        return grad_xyz1, grad_xyz2


chamfer_distance = ChamferDistanceFunction.apply
