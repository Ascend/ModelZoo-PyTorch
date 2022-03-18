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

PrecisionAtK, used for calculating the precision of results.

The PrecisionAtK class contains the implementation of the precision metric.
Its function is to evaluate results obtained using a certain model.
"""
import numpy as np
from metric.metric_at_k import MetricAtK
from metric.top_selector import TopSelector


class PrecisionAtK(MetricAtK):
    """PrecisionAtK class. Inherits the MetricAtK class.

    The PrecisionAtK is used to calculate the precision metric.

    Attributes:
        _top_selector: A class used to extract top results used in precision
            calculations.
    """
    def __init__(self, k):
        """Inits PrecisionAtK with its k value.
        k must be greater than 0.
        Raises:
            TypeError: The k value is not an integer or is not set.
            ValueError: The k value is smaller than 1.
        """
        super().__init__('Precision', k)
        self._top_selector = TopSelector()

    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the precision metric.

        Calculates the precision on the passed predicted and true values at k.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated precision value. The
            precision for one set of predictions is defined as follows:
            Precision@K = (# of recommended items @k that are relevant) /
             (# of recommended items @k)
            math::
            Precision@K = \\frac{n_{relevant & recommended}}{min(k,
            n_{recommended})}

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
                This is also raised if the output is not in [0,1].
        """
        self._check_input(y_true, y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self._check_args_numpy(y_pred, y_true)

        y_pred_binary = self._top_selector.find_top_k_binary(y_pred, self._k)
        y_true_binary = (y_true > 0)
        tmp = (np.logical_and(y_true_binary, y_pred_binary)
               .sum(axis=1)).astype(np.float32)
        precision = tmp / np.minimum(self._k,
                                     np.maximum(1, np.ones_like(y_pred_binary)
                                                .sum(axis=1)))
        result = precision.mean()
        if not (0 <= result <= 1):
            raise ValueError('The output of PrecisionAtK.evaluate \
                              must be in [0,1]')
        return result
