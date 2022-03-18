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

SingleObjectiveCDV, used to represent a Common
Descent Vector for single objective

The SingleObjectiveCDV class is mostly a wrapper
for a single objective loss
"""

from commondescentvector.common_descent_vector import CommonDescentVector


class SingleObjectiveCDV(CommonDescentVector):
    """ SingleObjectiveCDV class. Inherits from the CommonDescentVector
    class.

    SingleObjectiveCDV is used to return the common descent vector for
    a single objective

    Attributes:
        max_empirical_loss: A PyTorch Tensor loss representing the max empirical
            loss for the objective
        loss: A PyTorch Tensor loss representing the actual loss of the objective
    """

    def __init__(self, max_empirical_loss=None, normalized=False):
        """Inits SingleObjectiveCDV

        Args:
            max_empirical_loss: A Pytorch Tensor of size 1 containing the max empirical
                loss for the objective. Must be greater than 0 if set. Default: None
            normalized: Boolean, set to True if the output of get_descent_vector
            must be normalized. Default value: False
        Raises:
            ValueError: An error occured when max_empirical_loss is less than 0.
        """

        if max_empirical_loss is not None:
            if max_empirical_loss <= 0:
                raise ValueError('Argument: max_empirical_loss must be greater than 0')

        self.__max_empirical_loss = max_empirical_loss
        self.__normalized = normalized

    def get_descent_vector(self, losses, gradients=None):
        """Return the common descent vector for a single objective.
        Normalized if normalized has been set to True during initialization


        Args:
            losses: A numpy array containing 1 PyTorch Tensor representing the loss
            gradients: unused
        Return:
            loss: A pytorch tensor representing the loss
            alpha: Always returns 1 by this function
        Raise:
            TypeError: An error occured while accessing the argument - the
            argument is NoneType
        """

        if losses is None:
            raise TypeError('Argument: losses cannot be NoneType')
        if len(losses) != 1:
            raise ValueError('Argument: losses must be of size 1')

        if self.__normalized:
            if self.__max_empirical_loss is None:
                raise TypeError('max_empirical_loss must be set')
            else:
                return (losses[0]/self.__max_empirical_loss)
        else:
            return losses[0], 1

    def set_max_empirical_loss(self, max_empirical_loss):
        """Set the max empirical loss

        Args:
            max_empirical_loss: A PyTorch Tensor loss representing the max empirical
            loss for the objective
        """

        if max_empirical_loss <= 0:
            raise ValueError('Argument: max_empirical_loss must be greater than 0')

        self.__max_empirical_loss = max_empirical_loss

    def set_normalized(self, normalized):
        """Set if the CommonDescentVector should be normalized

        Args:
            normalized: True if the descent vector should be normalized
        """

        self.__normalized = normalized
