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

This test doesn't need any custom library or any data loading.
To run them just execute 'pytest'.
"""
from dataloader.mamo_dataset import MamoDataset
import pytest
import numpy as np

# Tests for 1-d data
test_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_output = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])

# Two objects for testing
mamoDataset1 = MamoDataset(test_input)
mamoDataset2 = MamoDataset(test_input, test_output)


# Testing the none input data error
def test_none_value_error_1d():
    input_data = None
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The input data is None, please give a valid input data.'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()

# Testing the length error


def test_length_value_error_1d():
    input_data = np.array([1, 2, 3])
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The length of the input data must match the length of the output data!'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()


# Testing the length method
def test_len_1d():
    assert len(test_input) == mamoDataset1.__len__()
    assert len(test_input) == mamoDataset2.__len__()
    assert len(test_output) == mamoDataset2.__len__()


# Testing the getitem method
def test_get_item_1d():
    # only input
    x, y = mamoDataset1.__getitem__(0)
    assert x == y == 1
    x, y = mamoDataset1.__getitem__(5)
    assert x == y == 6
    # input and output
    x, y = mamoDataset2.__getitem__(0)
    assert x == 1
    assert y == -1
    x, y = mamoDataset2.__getitem__(5)
    assert x == 6
    assert y == -6
    x, y = mamoDataset2.__getitem__(-5)
    assert x == 6
    assert y == -6


# Testing the none input data error
def test_get_item_1d_errors():
    with pytest.raises(IndexError, match=r'index [0-9]+ is out of bounds for dimension 0 with size [0-9]+'):
        x, y = mamoDataset1.__getitem__(55)
    with pytest.raises(IndexError):
        x, y = mamoDataset1.__getitem__(5.5)


# Tests for 2-d data
test_input_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [
    9, 10], [11, 12], [13, 14], [15, 16]])
test_output_2d = np.array([[-1, -2], [-3, -4], [-5, -6],
                           [-7, -8], [-9, -10], [-11, -12], [-13, -14], [-15, -16]])


mamoDataset1_2d = MamoDataset(test_input_2d)
mamoDataset2_2d = MamoDataset(test_input_2d, test_output_2d)


# Testing the length error
def test_length_value_error_2d():
    input_data = np.array([[1, 2], [3, 4], [5, 6]])
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The length of the input data must match the length of the output data!'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()


# Testing the length method
def test_len_2d():
    assert len(test_input_2d) == mamoDataset1_2d.__len__()
    assert len(test_input_2d) == mamoDataset2_2d.__len__()
    assert len(test_output_2d) == mamoDataset2_2d.__len__()


# Testing the getitem method
def test_get_item_2d():
    # only input
    x, y = mamoDataset1_2d.__getitem__(0)
    assert x[0] == y[0] == 1
    assert x[1] == y[1] == 2
    x, y = mamoDataset1_2d.__getitem__(5)
    assert x[0] == y[0] == 11
    assert x[1] == y[1] == 12
    # input and output
    x, y = mamoDataset2_2d.__getitem__(0)
    assert x[0] == 1
    assert x[1] == 2
    assert y[0] == -1
    assert y[1] == -2
    x, y = mamoDataset2_2d.__getitem__(5)
    assert x[0] == 11
    assert x[1] == 12
    assert y[0] == -11
    assert y[1] == -12
