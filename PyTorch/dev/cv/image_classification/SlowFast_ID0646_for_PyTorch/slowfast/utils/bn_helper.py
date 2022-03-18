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

"""bn helper."""

import itertools
import torch
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


@torch.no_grad()
def compute_and_update_bn_stats(model, data_loader, num_batches=200):
    """
    Compute and update the batch norm stats to make it more precise. During
    training both bn stats and the weight are changing after every iteration,
    so the bn can not precisely reflect the latest stats of the current model.
    Here the bn stats is recomputed without change of weights, to make the
    running mean and running var more precise.
    Args:
        model (model): the model using to compute and update the bn stats.
        data_loader (dataloader): dataloader using to provide inputs.
        num_batches (int): running iterations using to compute the stats.
    """

    # Prepares all the bn layers.
    bn_layers = [
        m
        for m in model.modules()
        if any(
            (
                isinstance(m, bn_type)
                for bn_type in (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                )
            )
        )
    ]

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    # Calculates the running iterations for precise stats computation.
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_square_mean = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for ind, (inputs, _, _) in enumerate(
        itertools.islice(data_loader, num_batches)
    ):
        # Forwards the model to update the bn stats.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].float().npu(f'npu:{NPU_CALCULATE_DEVICE}')
        else:
            inputs = inputs.npu(f'npu:{NPU_CALCULATE_DEVICE}')
        model(inputs)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            # $E(x^2) = Var(x) + E(x)^2$.
            cur_square_mean = bn.running_var + bn.running_mean ** 2
            running_square_mean[i] += (
                cur_square_mean - running_square_mean[i]
            ) / (ind + 1)

    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
        # Var(x) = $E(x^2) - E(x)^2$.
        bn.running_var = running_square_mean[i] - bn.running_mean ** 2
        # Sets the precise bn stats.
        bn.momentum = momentum_actual[i]
