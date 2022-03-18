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

DiversityAtK, used for calculating the diversity of results.

The DiversityAtK class contains the implementation of the diversity metric.
Its function is to evaluate results obtained using a certain model.
"""
import bottleneck as bn
import numpy as np
from scipy import spatial
from metric.metric_at_k import MetricAtK


class DiversityAtK(MetricAtK):
    """DiversityAtK class. Inherits the MetricAtK class.

    The DiversityAtK is used to calculate the diversity metric.
    """
    def __init__(self, k, diversity_vector):
        """Inits DiversityAtK with its k value and diversity vector.
        k must be greater than 0.
        Raises:
            TypeError: The k value is not an integer or is not set. The
                 diversity vector is not set or is of the incorrect type.
            ValueError: The k value is smaller than 1.  The diversity is empty.
        """
        super().__init__('Diversity', k)
        if diversity_vector is None:
            raise TypeError('Argument: diversity_vector must be set.')
        elif not isinstance(diversity_vector, np.ndarray):
            raise TypeError('Argument: diversity_vector must be a'
                            + ' numpy array.')
        elif diversity_vector.size == 0:
            raise ValueError('Argument: diversity_vector must not be an'
                             + ' empty array.')
        self._diversity_vector = diversity_vector

    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the diversity metric.

        Calculates the diversity on the passed predicted values at k.
        This method takes the top k predictions per input row and calculates
        the average dissimilarity between all pairs of these items.
        The dissimilarity is calculated using the values in the
        _diversity_vector.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated diversity value.
            The diversity is defined as follows:
            math::
            Diversity@K =
            \\frac{\\sum_{a \\in R_recommended@k}
            \\sum_{b \\in R_recommended@K}
            (1 - Similarity(a,b))}
            {N*(N-1)/2}
            From:
            https://pdfs.semanticscholar.org/
            b118/0953c16da28ec656d59e919ca6cf8a52b865.pdf

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
                Also occurs if passed arguments are not the same along the
                first dimension as axis 0 of the diversity vector stored in
                the object.
                This is also raised if the output is not in [0,1].
        """
        self._check_input(y_true, y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self._check_args_numpy(y_pred, y_true)
        if y_pred.shape[1] != self._diversity_vector.shape[0]:
            raise ValueError('Arguments must have axis 1 of the same size as\
            the axis 0 of the diversity vector.')

        rows = y_pred.shape[0]
        idx = bn.argpartition(-y_pred, self._k, axis=1)
        diversity = 0
        for a in range(rows):
            for i in range(self._k):
                if y_pred[a, idx[a][i]] <= 0:
                    continue
                for j in range(i):
                    if y_pred[a, idx[a][j]] <= 0:
                        continue
                    id1 = idx[a][i]
                    id2 = idx[a][j]
                    diversity += spatial.distance.cosine(
                                      self._diversity_vector[id1],
                                      self._diversity_vector[id2])
        result = diversity / (self._k*rows)
        if not (0 <= result <= 1):
            raise ValueError('The output of DiversityAtK.evaluate \
                              must be in [0,1]')
        return result
