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

from metric.top_selector import TopSelector
from metric.top_selector_torch import TopSelectorTorch
import numpy as np
import torch
import pytest

# Packages needed to run test:
# numpy
# bottleneck
# pytest


# Variables
k = 2
values = np.ones((2, 3))


# find_top_k_binary method raises TypeError if values is None.
def test_top_selector_find_top_k_binary_values_none():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(TypeError):
        ts.find_top_k_binary(None, k)
    with pytest.raises(TypeError):
        tst.find_top_k_binary(None, k)


# find_top_k_binary method raises TypeError if k is None.
def test_top_selector_find_top_k_binary_k_none():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(TypeError):
        ts.find_top_k_binary(values, None)
    with pytest.raises(TypeError):
        tst.find_top_k_binary(torch.from_numpy(values), None)


# find_top_k_binary method raises TypeError if k is not int.
def test_top_selector_find_top_k_binary_k_not_int():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(TypeError):
        ts.find_top_k_binary(values, 2.5)
    with pytest.raises(TypeError):
        tst.find_top_k_binary(torch.from_numpy(values), 2.5)


# find_top_k_binary method raises ValueError if values is not 2D.
def test_top_selector_find_top_k_binary_values_not_2D():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(ValueError):
        ts.find_top_k_binary(np.empty((2, 3, 3)), k)
    with pytest.raises(ValueError):
        ts.find_top_k_binary(np.empty(3), k)
    with pytest.raises(ValueError):
        tst.find_top_k_binary(torch.from_numpy(np.empty((2, 3, 3))), k)
    with pytest.raises(ValueError):
        tst.find_top_k_binary(torch.from_numpy(np.empty(3)), k)


# find_top_k_binary method raises ValueError if k is smaller than 0.
def test_top_selector_find_top_k_binary_k_negative():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(ValueError):
        ts.find_top_k_binary(values, -1)
    with pytest.raises(ValueError):
        tst.find_top_k_binary(torch.from_numpy(values), -1)


# find_top_k_binary method raises ValueError if k is
# larger than values.shape[1].
def test_top_selector_find_top_k_binary_k_larger_than_length():
    ts = TopSelector()
    tst = TopSelectorTorch()
    with pytest.raises(ValueError):
        ts.find_top_k_binary(values, values.shape[1])
    with pytest.raises(ValueError):
        tst.find_top_k_binary(torch.from_numpy(values), values.shape[1])


# find_top_k_binary correct case selecting top 2 from every row.
def test_top_selector_find_top_k_binary_correct_case():
    ts = TopSelector()
    tst = TopSelectorTorch()
    values = np.array([[0.4, 0.7, 0.1], [0.1, 0.3, 0.6], [0.02, 0.25, 0.2]])
    ground_truth = np.array([[True, True, False], [False, True, True],
                             [False, True, True]])
    assert np.all(ts.find_top_k_binary(values, k) == ground_truth)
    assert torch.equal(tst.find_top_k_binary(torch.from_numpy(values), k),
                       torch.from_numpy(ground_truth))


# find_top_k_binary correct case returning all False when all values are 0.
def test_top_selector_find_top_k_binary_correct_case_zeros():
    ts = TopSelector()
    tst = TopSelectorTorch()
    values = np.zeros((3, 3))
    ground_truth = np.zeros((3, 3), dtype=bool)
    assert np.all(ts.find_top_k_binary(values, k) == ground_truth)
    assert torch.equal(tst.find_top_k_binary(torch.from_numpy(values), k),
                       torch.from_numpy(ground_truth))


# find_top_k_binary correct case returning 1 per row instead of 2 when there is
# only 1 non-zero value in the given row.
def test_top_selector_find_top_k_binary_correct_case_some_zeros():
    ts = TopSelector()
    tst = TopSelectorTorch()
    values = np.array([[0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.02, 0.0, 0.0]])
    ground_truth = np.array([[True, False, False], [False, True, False],
                             [True, False, False]])
    assert np.all(ts.find_top_k_binary(values, k) == ground_truth)
    assert torch.equal(tst.find_top_k_binary(torch.from_numpy(values), k),
                       torch.from_numpy(ground_truth))
