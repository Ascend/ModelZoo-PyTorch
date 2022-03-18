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
"""Copyright 漏 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

from metric.recall_at_k import RecallAtK
from dataloader.mamo_dataset import MamoDataset
from validator import Validator
from torch.utils.data import DataLoader
from tests.integration.mocks.mock_models import MockAllZeros
from tests.integration.mocks.mock_models import MockNoChange
from tests.integration.mocks.mock_models import MockOpposite
from tests.integration.mocks.mock_models import MockShiftRightByOne
from tests.integration.mocks.mock_loss import MSELoss
import numpy as np

# Packages needed to run test:
# os
# numpy
# torch
# pytest

# Variables
# Mock dataset
input_data = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
input_data = input_data.astype(float)


# Test demonstrating mock all zeros model to show missing
# metrics or missing objectives
def test_validator_mock_all_zeros_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_all_zeros = Validator(MockAllZeros(), mock_dataloader,
                            [RecallAtK(1)], None)
    results = v_all_zeros.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    assert(results[1] == [])
    v_all_zeros = Validator(MockAllZeros(), mock_dataloader,
                            None, [MSELoss()])
    results = v_all_zeros.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0] == [])
    assert isinstance(results[1], list)
    mse = np.mean(input_data)
    assert(round(results[1][0], 2) == round(mse, 2))
    assert(round(v_all_zeros.combine_objectives(results[1]), 2)
           == round(mse, 2))


# Test demonstrating mock no change model
# Recall is 0 as we are recommending already chosen elements
def test_validator_mock_no_change_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_no_change = Validator(MockNoChange(), mock_dataloader,
                            [RecallAtK(1)], [MSELoss()])
    results = v_no_change.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    # Removing chosen elements -so:
    mse = np.mean(input_data)
    assert(round(results[1][0], 2) == round(mse, 2))
    assert(round(v_no_change.combine_objectives(results[1]), 2)
           == round(mse, 2))


# Test demonstrating mock opposite model
def test_validator_mock_opposite_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_opposite = Validator(MockOpposite(), mock_dataloader,
                           [RecallAtK(1)], [MSELoss()])
    results = v_opposite.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    assert(results[1][0] == 1)
    assert(v_opposite.combine_objectives(results[1]) == 1)


# Test demonstrating mock shift right by one model
def test_validator_mock_shift_right_by_one_model():
    mock_dataset = MamoDataset(input_data, np.roll(input_data.copy(),
                               shift=1, axis=1))
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_shift_right = Validator(MockShiftRightByOne(), mock_dataloader,
                              [RecallAtK(1)], [MSELoss()])
    results = v_shift_right.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 1)
    assert isinstance(results[1], list)
    assert(results[1][0] == 0)
    assert(v_shift_right.combine_objectives(results[1]) == 0)
