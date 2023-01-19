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
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['pixel_group'])


def pixel_group(
    score: Union[np.ndarray, Tensor],
    mask: Union[np.ndarray, Tensor],
    embedding: Union[np.ndarray, Tensor],
    kernel_label: Union[np.ndarray, Tensor],
    kernel_contour: Union[np.ndarray, Tensor],
    kernel_region_num: int,
    distance_threshold: float,
) -> List[List[float]]:
    """Group pixels into text instances, which is widely used text detection
    methods.

    Arguments:
        score (np.array or torch.Tensor): The foreground score with size hxw.
        mask (np.array or Tensor): The foreground mask with size hxw.
        embedding (np.array or torch.Tensor): The embedding with size hxwxc to
            distinguish instances.
        kernel_label (np.array or torch.Tensor): The instance kernel index with
            size hxw.
        kernel_contour (np.array or torch.Tensor): The kernel contour with
            size hxw.
        kernel_region_num (int): The instance kernel region number.
        distance_threshold (float): The embedding distance threshold between
            kernel and pixel in one instance.

    Returns:
        list[list[float]]: The instance coordinates and attributes list. Each
        element consists of averaged confidence, pixel number, and coordinates
        (x_i, y_i for all pixels) in order.
    """
    assert isinstance(score, (torch.Tensor, np.ndarray))
    assert isinstance(mask, (torch.Tensor, np.ndarray))
    assert isinstance(embedding, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_contour, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_region_num, int)
    assert isinstance(distance_threshold, float)

    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if isinstance(kernel_label, np.ndarray):
        kernel_label = torch.from_numpy(kernel_label)
    if isinstance(kernel_contour, np.ndarray):
        kernel_contour = torch.from_numpy(kernel_contour)

    if torch.__version__ == 'parrots':
        label = ext_module.pixel_group(
            score,
            mask,
            embedding,
            kernel_label,
            kernel_contour,
            kernel_region_num=kernel_region_num,
            distance_threshold=distance_threshold)
        label = label.tolist()
        label = label[0]
        list_index = kernel_region_num
        pixel_assignment = []
        for x in range(kernel_region_num):
            pixel_assignment.append(
                np.array(
                    label[list_index:list_index + int(label[x])],
                    dtype=np.float))
            list_index = list_index + int(label[x])
    else:
        pixel_assignment = ext_module.pixel_group(score, mask, embedding,
                                                  kernel_label, kernel_contour,
                                                  kernel_region_num,
                                                  distance_threshold)
    return pixel_assignment
