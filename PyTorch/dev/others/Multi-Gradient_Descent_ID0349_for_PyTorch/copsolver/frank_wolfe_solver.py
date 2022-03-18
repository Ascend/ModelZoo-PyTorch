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

FrankWolfeSolver, used to solver the Quadratic
Constrained Optimization Problem for 2 or more
gradients numerically.

The FrankWolfe class contains the implementation of
the numerical QCOP Solver for 2 or more gradients.
"""
import numpy as np
from copsolver.copsolver import COPSolver
import logging


class FrankWolfeSolver(COPSolver):
    """ FrankWolfeSolver class. Inherits from the COPSolver
    class.

    FrankWolfeSolver is used to calculate the numerical solutions
    for the QCOP for 2 or more gradients

    Attributes:
    max_iter: max number of iterations for the algorithm
    min_change: minimum change stopping criterion. The algorithms
    stop when the difference between iterations is lower than
    min_change
    """

    def __init__(self, max_iter=100, min_change=1e-3):
        """Inits FrankWolfeSolver with hyperparameters values

        Args:
            max_iter: maximum number of iterations. Must be <= 1.
                default value is 100.
            min_change: minimum change stopping criterion. Must be < 0
                default value is 1e-3
        """

        if max_iter < 0:
            raise ValueError('Argument, max_iter must be positive')
        if min_change < 0:
            raise ValueError('Arguement: min_change must be positive')

        self._max_iter = max_iter
        self._min_change = min_change
        super().__init__()

    def solve(self, gradients):
        """Solves the Constrained Optimization Problem

            Given the gradients, compute numerically the alphas for the COP
             and returns them in a list
        Args:
            gradients: numpy array of gradients from the models. gradients
            are numpy array of floats
        Returns:
            A numpy array of floats in [0,1] repreenting
            coefficients associated to the gradients
        Raises:
            ValueError: Error occured when checking the dimensions
                of gradients
            TypeError: An error occured while accessing the argument - one
                of the arguments is NoneType


        source: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
        (page 5, alg 2)
        """

        if gradients is None:
            raise TypeError('Argument: gradients must be set.')
        for gradient in gradients:
            if(len(gradient) != len(gradients[0])):
                raise ValueError('Argument: gradients must have the same length')

        # number of objectives
        n = len(gradients)

        if (n == 2):
            logging.warning('Arugment: There are 2 gradients. Use the AnalyticalSolver instead.')

        # if there is only 1 gradient, alpha is equal to 1
        if (n == 1):
            return np.ones(1)

        # initialize alphas
        alphas_list = np.ones(n) / n

        # precompute gradient products
        M = gradients @ gradients.T

        for i in range(self._max_iter):
            # find objective whose gradient gives the smallest product
            # (min row in M)
            min_objective = np.argmin(alphas_list @ M)

            # find min gamma
            # v1 = alphas_list @ gradients   <- combined gradient
            # v2 = gradients[min_objective]  <- min gradient
            v1_v1 = alphas_list @ M @ alphas_list
            v1_v2 = alphas_list @ M[min_objective]
            v2_v2 = M[min_objective, min_objective]
            min_gamma = self.__min_norm_2(v1_v1, v1_v2, v2_v2)

            # update alpha
            new_alphas_list = min_gamma * alphas_list
            new_alphas_list[min_objective] += 1 - min_gamma

            # if update is smaller than min_change stop
            if np.sum(np.abs(new_alphas_list - alphas_list)) \
                    < self._min_change:
                return new_alphas_list
            else:
                alphas_list = new_alphas_list

            return alphas_list

    def __min_norm_2(self, v1_v1, v1_v2, v2_v2):
        r"""Helper function for the Frank Wolve Solver, compute the min alpha of
        the norm squared between two gradients

        .. math::

            \alpha = \frac{(\overline{\theta} - \theta)^{T} \overline{\theta}}
                          {\|\theta - \overline{\theta}\|_{2}^{2}}

            where v1_v1 = \theta^{T}\theta
                v1_v2 = \theta^{T}\overline{\theta}
                v2_v2 = \overline{\theta}^{T}\overline{\theta}

        source: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
        (page 5, alg 1)
        """
        if v1_v1 <= v1_v2:
            return 1.0
        if v2_v2 <= v1_v2:
            return 0.0
        # calculate and return alpha
        return (v2_v2 - v1_v2) / (v1_v1 + v2_v2 - 2 * v1_v2)
