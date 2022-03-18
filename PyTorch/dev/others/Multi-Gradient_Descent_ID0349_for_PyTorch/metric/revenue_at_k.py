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

RevenueAtK, used for calculating the revenue of results.

The RevenueAtK class contains the implementation of the revenue metric.
Its function is to evaluate results obtained using a certain model.
"""
import numpy as np
from metric.metric_at_k import MetricAtK
from metric.top_selector import TopSelector


class RevenueAtK(MetricAtK):
    """RevenueAtK class. Inherits the MetricAtK class.

    The RevenueAtK is used to calculate the revenue metric.

    Attributes:
        _revenue: The revenue of items on which the metric will be calculated
        _top_selector: A class used to extract top results used in revenue
            calculations.
    """
    def __init__(self, k, revenue):
        """Inits RevenueAtK with its k value and revenue.
        k must be greater than 0.
        Raises:
            TypeError: The k value is not an integer or is not set. The revenue
                is not set or is of the incorrect type. The revenue
                contains something other than floats and/or integers
            ValueError: The k value is smaller than 1. The revenue is empty.
        """
        super().__init__('Revenue', k)
        if revenue is None:
            raise TypeError('Argument: revenue must be set.')
        elif not isinstance(revenue, np.ndarray) or revenue.ndim != 1:
            raise TypeError('Argument: revenue must be a 1D numpy array.')
        elif revenue.size == 0:
            raise ValueError('Argument: revenue must not be an empty array.')
        elif not all(isinstance(i, (np.floating, float, np.integer, int))
                     for i in revenue):
            raise TypeError('All elements of argument: revenue must be'
                            + ' of type int or float.')
        self._revenue = revenue
        self._top_selector = TopSelector()

    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the revenue metric.

        Calculates the revenue on the passed predicted and true values at k.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated revenue value. The revenue
            is defined as the revenue of the relevant predicted values.
            math:: Revenue@K = \\sum_{i \\in R_{relevant & recommended}}value_i

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
                Also occurs if passed arguments are not the same along the
                first dimension as the revenue stored in the object.
        """
        self._check_input(y_true, y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self._check_args_numpy(y_pred, y_true)
        if y_pred.shape[1] != len(self._revenue):
            raise ValueError('Arguments must have axis 1 of the same size as\
            the revenue.')

        y_pred_binary = self._top_selector.find_top_k_binary(y_pred, self._k)
        y_true_binary = (y_true > 0)
        tmp = np.logical_and(y_true_binary, y_pred_binary)
        revenue = 0
        for i in range(tmp.shape[0]):
            revenue += np.sum(self._revenue[tmp[i]])
        return revenue / float(tmp.shape[0])
