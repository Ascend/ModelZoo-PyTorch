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

MultiObjectiveCDV, used to represent a Common
Descent Vector for multiple objectives

The MultiObjectiveCDV class contains the implementation
of the Common Descent Vector for multiple objectives
"""

from commondescentvector.common_descent_vector import CommonDescentVector
import logging
import time


class MultiObjectiveCDV(CommonDescentVector):
    """ MultiObjectiveCDV class. Inherits from the CommonDescentVector
    class.

    MultiObjectiveCDV is used to return the common descent vector for
    a single objective

    Attributes:
        max_empirical_loss: A numpy array of PyTorch Tensor losses representing the max empirical
            loss for each of the objectives
        copsolver: A COPSolver used for solving the Quadratic Constraint Optimization Problem
        losses: A numpy array of PyTorch Tensor losses representing the actual loss of the objective
        gradients: A multidimensional numpy array representing the gradients of the model
        normalized: Boolean, True if the output of get_descent_vector must be normalized.
    """

    def __init__(self, copsolver, max_empirical_losses=None, normalized=False):
        """Inits MultiObjectiveCDV

        Args:
            copsolver: COPSolver object used to solve the QCOP
            max_empirical_losses: A numpy array of PyTorch Tensor losses representing the max empirical
            loss for each of the objectives
            normalized: Boolean, set to True if the output of get_descent_vector
            must be normalized. Default value: False
        """

        if copsolver is None:
            raise TypeError('Argument: copsolver cannot be NoneType')

        if max_empirical_losses is not None:
            if any(loss <= 0 for loss in max_empirical_losses):
                raise ValueError('Argument: each element of max_empirical loss should be greater than 0')

        self.__max_empirical_losses = max_empirical_losses
        self.__copsolver = copsolver
        self.__normalized = normalized

    def get_descent_vector(self, losses, gradients):
        """Compute and return the Common Descent Vector. Normalize it by the max_empirical
            loss if normalized is set to True.

        Args:
            losses: A numpy array of PyTorch Tensor losses representing the loss
            for each of the objectives.
            gradients: A numpy array of floats representing the gradients for
            each of the objectives

        Returns:
            loss: A PyTorch Tensor loss representing the Common Descent Vector
            alphas: a list of float representing the alphas from the COPSolver
        Raises:
            TypeError: An error occured when there is no objective.
        """

        if losses is None:
            raise TypeError('Argument: losses cannot be NoneType')
        if gradients is None:
            raise TypeError('Argument: gradients cannot be NoneType')
        if len(losses) == 0:
            raise ValueError('Argument: losses cannot be empty')
        if len(gradients) == 0:
            raise ValueError('Argument: gradients cannot be empty')
        if len(gradients) != len(losses):
            raise ValueError('Argument: losses and gradients array must have the same length')

        if self.__normalized:
            if self.__max_empirical_losses is None:
                raise TypeError('Argument: max_empirical_loss must be set')
            if len(losses) != len(self.__max_empirical_losses):
                raise ValueError('Argument: losses must have the same length as max_empirical_losses')
            if len(gradients) != len(self.__max_empirical_losses):
                raise ValueError('Argument: gradients must have the same length as max_empirical_losses')

            alphas = self.__solve_qcop(gradients, self.__normalized)

            for i in range(len(losses)):
                if (i == 0):
                    loss = (alphas[i]/self.__max_empirical_losses[i]) * losses[i]
                else:
                    loss += (alphas[i]/self.__max_empirical_losses[i]) * losses[i]
        else:

            alphas = self.__solve_qcop(gradients, self.__normalized)

            for i in range(len(losses)):
                if (i == 0):
                    loss = alphas[i] * losses[i]
                else:
                    loss += alphas[i] * losses[i]

        return loss, alphas

    def set_max_empirical_losses(self, max_empirical_losses):
        """Set the max empirical loss

        Args:
            max_empirical_loss: A numpy array of PyTorch Tensor loss representing the max empirical
            loss for the objective
        """

        if any(loss <= 0 for loss in max_empirical_losses):
            raise ValueError('Argument: each element of max_empirical loss should be greater than 0')

        self.__max_empirical_losses = max_empirical_losses

    def set_normalized(self, normalized):
        """Set if the CommonDescentVector should be normalized

        Args:
            normalized: True if the descent vector should be normalized
        """

        self.__normalized = normalized

    def __solve_qcop(self, gradients, normalized):
        """Solve the QCOP

        Args:
            normalized: True if the descent vector should be normalized
            gradients: A numpy array of floats representing the gradients for
            each of the objectives

        Returns:
            alphas: list of floats represting the alphas correspond to the weights
            assigned by the QCOP
        """
        logging.debug('CDV: Solving QCOP...')
        start = time.process_time()
        try:
            if (normalized):
                for i in range(len(gradients)):
                    gradients[i] /= self.__max_empirical_losses[i]
            alphas = self.__copsolver.solve(gradients)
        except Exception as err:
            logging.exception(err)
        logging.debug('QCOP completed in: %.4f' % (time.process_time() - start))

        return alphas
