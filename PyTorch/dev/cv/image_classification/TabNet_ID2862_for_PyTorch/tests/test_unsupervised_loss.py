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
import numpy as np
import torch
import pytest
from py_tabnet.metrics import UnsupervisedLoss, UnsupervisedLossNumpy
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


@pytest.mark.parametrize(
    "y_pred,embedded_x,obf_vars",
    [
        (
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.choice([0, 1], size=(20, 100), replace=True)
        ),
        (
            np.random.uniform(low=-2, high=2, size=(30, 50)),
            np.ones((30, 50)),
            np.random.choice([0, 1], size=(30, 50), replace=True)
        )
    ]
)
def test_equal_losses(y_pred, embedded_x, obf_vars):
    numpy_loss = UnsupervisedLossNumpy(
        y_pred=y_pred,
        embedded_x=embedded_x,
        obf_vars=obf_vars
    )

    torch_loss = UnsupervisedLoss(
        y_pred=torch.tensor(y_pred, dtype=torch.float64),
        embedded_x=torch.tensor(embedded_x, dtype=torch.float64),
        obf_vars=torch.tensor(obf_vars, dtype=torch.float64)
    )

    assert np.isclose(numpy_loss, torch_loss.detach().numpy())
