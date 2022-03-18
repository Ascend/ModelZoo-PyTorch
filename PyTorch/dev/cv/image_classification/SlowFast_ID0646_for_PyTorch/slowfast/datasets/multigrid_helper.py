#!/usr/bin/env python3
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Helper functions for multigrid training."""

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR >= 1 and TORCH_MINOR >= 8:
    _int_classes = int
else:
    from torch._six import int_classes as _int_classes


class ShortCycleBatchSampler(Sampler):
    """
    Extend Sampler to support "short cycle" sampling.
    See paper "A Multigrid Method for Efficiently Training Video Models",
    Wu et al., 2019 (https://arxiv.org/abs/1912.00998) for details.
    """

    def __init__(self, sampler, batch_size, drop_last, cfg):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.drop_last = drop_last

        bs_factor = [
            int(
                round(
                    (
                        float(cfg.DATA.TRAIN_CROP_SIZE)
                        / (s * cfg.MULTIGRID.DEFAULT_S)
                    )
                    ** 2
                )
            )
            for s in cfg.MULTIGRID.SHORT_CYCLE_FACTORS
        ]

        self.batch_sizes = [
            batch_size * bs_factor[0],
            batch_size * bs_factor[1],
            batch_size,
        ]

    def __iter__(self):
        counter = 0
        batch_size = self.batch_sizes[0]
        batch = []
        for idx in self.sampler:
            batch.append((idx, counter % 3))
            if len(batch) == batch_size:
                yield batch
                counter += 1
                batch_size = self.batch_sizes[counter % 3]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        avg_batch_size = sum(self.batch_sizes) / 3.0
        if self.drop_last:
            return int(np.floor(len(self.sampler) / avg_batch_size))
        else:
            return int(np.ceil(len(self.sampler) / avg_batch_size))
