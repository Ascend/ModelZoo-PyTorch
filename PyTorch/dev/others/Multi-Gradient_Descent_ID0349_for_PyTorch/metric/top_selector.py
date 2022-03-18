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

TopSelector class, used selecting top samples.

The TopSelector class is a helper class used by different metric
implementations to select the top values from arrays. This logic is extracted
into a class of its own to increase reusability and simplify testing.
this class uses NumPy and will slowly be fazed out.
"""
import bottleneck as bn
import numpy as np


class TopSelector:
    """TopSelector class."""

    def find_top_k_binary(self, values, k):
        """Finds the top k values for each row of a matrix and returns a binary
        mask on their positions.

        The method masks the k input values with the highest numerical value in
        every row of the input 2D numpy array.

        Args:
            values: A numpy array of values.
            k: An integer that denotes the number of values to obtain from the
                ranking of the values. The method masks the k values with the
                highest scores.

        Returns:
            A binary mask in the form of a numpy 2D array that outputs the
            top k values per row from the input values.
            For example:

            values = array([[0.5, 0.7, 0.3],
                                 [0.4, 0.1, 0.7]])
            k = 2
            find_top_k_binary returns:

            array([[ True, True, False],
                   [ True, False, True]])

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                values argument. It is not a 2D array. Or if k is smaller
                than 0.
        """
        if values is None:
            raise TypeError('Argument: values must be set.')
        if k is None:
            raise TypeError('Argument: k must be set.')
        if not isinstance(k, (int, np.integer)):
            raise TypeError('Argument: k must be an integer.')
        if values.ndim != 2:
            raise ValueError('Argument: values must be a 2D array.')
        if k < 1:
            raise ValueError('Argument: k cannot be negative.')
        if k >= values.shape[1]:
            raise ValueError('Argument: k cannot be larger than\
                             values.shape[1]')

        rows = values.shape[0]
        idx = bn.argpartition(-values, k, axis=1)
        values_binary = np.zeros_like(values, dtype=bool)
        values_binary[np.arange(rows)[:, np.newaxis], idx[:, :k]] = True
        values_binary[np.where(values <= 0)] = False
        return values_binary
