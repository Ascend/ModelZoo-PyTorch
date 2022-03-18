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

HitRatioAtK, used for calculating the hit ratio of results.

The HitRatioAtK class contains the implementation of the hit ratio metric.
Its function is to evaluate results obtained using a certain model.
"""
import numpy as np
from metric.metric_at_k import MetricAtK
from metric.top_selector import TopSelector


class HitRatioAtK(MetricAtK):
    """HitRatioAtK class. Inherits the MetricAtK class.

    The HitRatioAtK is used to calculate the hit ratio metric.

    Attributes:
        _top_selector: A class used to extract top results used in hit ratio
            calculations.
    """
    def __init__(self, k):
        """Inits HitRatioAtK with its k value.
        k must be greater than 0.
        Raises:
            TypeError: The k value is not an integer or is not set.
            ValueError: The k value is smaller than 1.
        """
        super().__init__('Hit Ratio', k)
        self._top_selector = TopSelector()

    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the hit ratio metric.

        Calculates the hit ratio on the passed predicted and true values at k.

        Args:
            y_true: A PyTorch tensor of true values. Only one value per row
                    can be > 0!
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated hit ratio value. The hit
            ratio is defined as follows:
            math::
            HR@K = \\frac{Number of Hits @ K}{Number of Ground Truth Items(=1)}
            This is then averaged over all sets of predictions/ground truths
            (users).
            From:
            https://www.comp.nus.edu.sg/~kanmy/papers/cikm15-trirank-cr.pdf

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
                If y_true has more than one true value per row this error
                is raised. This is also raised if the output is not in [0,1].
        """
        self._check_input(y_true, y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self._check_args_numpy(y_pred, y_true)

        # Check only one ground truth value = 1 per row in y_true.
        y_true[y_true > 0] = 1
        y_true[y_true < 0] = 0
        for x in np.sum(y_true, axis=1):
            if x != 1:
                raise ValueError('Incorrect format of argument: y_true. \
                                  Input must have only one true value \
                                  per row.')

        y_pred_binary = self._top_selector.find_top_k_binary(y_pred, self._k)
        y_true_binary = (y_true > 0)
        result = (np.logical_and(y_true_binary, y_pred_binary)
                  .sum(axis=1)).astype(np.float32).mean()
        if not (0 <= result <= 1):
            raise ValueError('The output of HitRatioAtK.evaluate \
                              must be in [0,1]')
        return result
