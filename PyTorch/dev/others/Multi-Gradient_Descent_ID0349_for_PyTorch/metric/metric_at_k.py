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

Abstract Metric class, used for evaluating results.

The abstract Metric class contains a basic skeleton and some implementation
details that will be shared among its children classes. Its function is to
evaluate results obtained using a certain model.
"""
from abc import ABC, abstractmethod
import torch


class MetricAtK(ABC):
    """Abstract MetricAtK class.

    The Abstract MetricAtK class represents the parent class that is inherited
    if all concrete metric implementations.

    Attributes:
        _name: A string indicating the name of the metric.
        _k: An integer denoting the first N items upon which to calculate
            the given metric.
    """
    def __init__(self, name, k):
        """Inits MetricAtK with its name and k value.
        Raises:
            TypeError: The k value is not an integer or is not set.
            ValueError: The k value is smaller than 1.
        """
        if k is None:
            raise TypeError('Argument: k must be set.')
        elif not isinstance(k, int):
            raise TypeError('Argument: k must be an integer.')
        elif k <= 0:
            raise ValueError('Argument: k must be larger than 0.')
        self._name = name
        self._k = k
        super().__init__()

    def get_name(self):
        """Returns the name of the MetricAtK class."""
        return self._name + ' at ' + str(self._k)

    def _check_args_numpy(self, y_pred, y_true):
        """Evaluates the validity of the passed arguments.

        Checkes the validity of the arguments. Used to assert correctness of
        input for evaluate() method.

        Args:
            y_pred: A 2D numpy array values.
            y_true: A 2D numpy array values.

        Returns:
            Nothing in case of correct argument format.
        Raises:
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
        """
        if y_pred.ndim != 2:
            raise ValueError('Argument: y_pred must be a 2D array.')
        if y_true.ndim != 2:
            raise ValueError('Argument: y_true must be a 2D array.')
        if y_pred.shape != y_true.shape:
            raise ValueError('Arguments must be of the same shape.')

    def _check_args_torch(self, y_pred, y_true):
        """Evaluates the validity of the passed arguments.

        Checkes the validity of the arguments. Used to assert correctness of
        input for evaluate() method.

        Args:
            y_pred: A 2D PyTorch tensor of values.
            y_true: A 2D PyTorch tensor of values.

        Returns:
            Nothing in case of correct argument format.
        Raises:
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D tensors,
                or they are 2D but of different sizes along those dimensions.
        """
        if y_pred.ndimension() != 2:
            raise ValueError('Argument: y_pred must be a 2D tensor.')
        if y_true.ndimension() != 2:
            raise ValueError('Argument: y_true must be a 2D tensor.')
        if y_pred.shape != y_true.shape:
            raise ValueError('Arguments must be of the same shape.')

    def _check_input(self, y_true, y_pred):
        """Evaluates the input arguments.

        Used to assert correctness of input types for evaluate() method.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Nothing in case of correct argument format.
        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType or not a PyTorch tensor.
        """
        if y_true is None:
            raise TypeError('Argument: y_true must be set.')
        if y_pred is None:
            raise TypeError('Argument: y_pred must be set.')
        if not torch.is_tensor(y_true):
            raise TypeError('Argument: y_true must be a PyTorch tensor.')
        if not torch.is_tensor(y_pred):
            raise TypeError('Argument: y_pred must be a PyTorch tensor.')

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the implemented metric.

        Calculates the implemented metric on the passed predicted and true
        values at k.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated metric value, currently
            unimplemented.
        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments.
        """
        pass
