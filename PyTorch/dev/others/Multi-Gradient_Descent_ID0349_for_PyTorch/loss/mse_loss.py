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

MSELoss, used for calculating the Mean Squared Error.

The MSELoss class contains the implementation of the MSE.
MSE is a commonly used loss function, for example for regression.
"""
from loss.loss_class import Loss
import torch


class MSELoss(Loss):
    """MSELoss class. Inherits the Loss class.

    The MSELoss is used to calculate the MSE loss.
    """
    def __init__(self):
        """Inits MSELoss."""
        super().__init__('MSELoss')

    def compute_loss(self, y_true, output_model):
        """Calculates the MSE loss between the predictions and ground truths.

        Calculates the MSE on the passed predicted and true values.
        The loss is the mean of the squared differences between true
        and predicted values.

        Args:
            y_true: A pytorch tensor of ground truth values.
            output_model: A pytorch tensor of predicted values.

        Returns:
            Will return a float with the calculated MSE value.
            math::
            MSE = \\frac{\\sum_{i=1}^{n}(Y_i - \\hat{Y_i})^2}{n}

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType, or not a pytorch tensor.
            ValueError: An error occured when checking the dimensions of the
                y_true and output_model arguments if they are not equal.
                This is also raised if the output is not in [0,1].
        """
        if output_model is None:
            raise TypeError('Argument: output_model must be set.')
        if y_true is None:
            raise TypeError('Argument: y_true must be set.')
        if not isinstance(output_model, torch.Tensor):
            raise TypeError('Argument: output_model must be a pytorch tensor.')
        if not isinstance(y_true, torch.Tensor):
            raise TypeError('Argument: y_true must be a pytorch tensor.')
        # check if good dimensions between predictions and ground truth
        self.__check_dim_pred_gt__(output_model, y_true)

        result = ((output_model - y_true) ** 2).sum()\
            / output_model.data.nelement()
        if not (0 <= result <= 1):
            raise ValueError('The output of MSELoss.compute_loss \
                              must be in [0,1]')
        return result
