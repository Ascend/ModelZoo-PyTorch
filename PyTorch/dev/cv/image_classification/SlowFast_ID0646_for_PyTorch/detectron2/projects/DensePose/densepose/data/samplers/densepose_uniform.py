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

import random
import torch

from .densepose_base import DensePoseBaseSampler


class DensePoseUniformSampler(DensePoseBaseSampler):
    """
    Samples DensePose data from DensePose predictions.
    Samples for each class are drawn uniformly over all pixels estimated
    to belong to that class.
    """

    def __init__(self, count_per_class: int = 8):
        """
        Constructor

        Args:
          count_per_class (int): the sampler produces at most `count_per_class`
              samples for each category
        """
        super().__init__(count_per_class)

    def _produce_index_sample(self, values: torch.Tensor, count: int):
        """
        Produce a uniform sample of indices to select data

        Args:
            values (torch.Tensor): an array of size [n, k] that contains
                estimated values (U, V, confidences);
                n: number of channels (U, V, confidences)
                k: number of points labeled with part_id
            count (int): number of samples to produce, should be positive and <= k

        Return:
            list(int): indices of values (along axis 1) selected as a sample
        """
        k = values.shape[1]
        return random.sample(range(k), count)
