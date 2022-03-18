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
"""

from loss.loss_class import Loss
import torch
import torch.nn.functional as F


class VAELoss(Loss):
    """
    Implementation of the Variational Auto-Encoder Loss for the VAE model, inherits from Loss.

    There is multiple different instance of that class that can be created.

    For Example:
        -relevance_loss :  This loss measures the 'relevance' of the predictions between users and items.
                           This is used to predict a vector of binary vector that predict
                           if a user will interact with an item.

        -price_loss : This is used to predict the price of item that the user will interact with.

        More precisely:
        Suppose the y_true are your labels and weighted_vector are the prices of every item, you could have:
        relevance_loss = VAELoss(y_pred, y_true, mean, log_variance, anneal)
        weighted_vector = price_vector
        price_loss = VAELoss(y_pred, y_true, mean, log_variance, anneal, weighted_vector)

    Given the predictions of the model and the ground-truth, computes the loss (float)


    """

    def __init__(self, name='VAELoss', weighted_vector=None):
        """
        weighted_vector: vector of weights of every item, default None. Could be for example the prices of
                         every item (price_vector) or novelty_vector etc.
        """
        super().__init__(name)
        self.weighted_vector = weighted_vector

    def compute_loss(self, y_true, output_model, anneal=1.0):
        """
        Compute the loss of VAE model that sums two componants:
            -reconstruction loss which needs predictions and ground-truth.
            -regularization loss which needs mean, log_variance and the anneal factor (to faster convergence)

        Attributes:
        y_true: ground-truth labels
        output_model = (y_pred, mean, log_variance) where
                    y_pred: prediction of the model
                    mean: mean of distribution function of the encoder
                    log_variance: log variance of the distribtion function of the encoder
        anneal: annealing value for the regularization loss, default 1.0

        """

        # for vae, output_model = (y_pred, mean, log_variance)
        y_pred = output_model[0]
        mean = output_model[1]
        log_variance = output_model[2]

        # check if good dimensions between y_pred and y_true
        self.__check_dim_pred_gt__(y_pred, y_true)
        # check if mean or log_variance are not None
        self.__check_is_mean_var__(mean, log_variance)

        # calculate the reconstruction loss
        if(self.weighted_vector is not None):
            # reconstruction loss if user provides a weighted vector
            BCE = -torch.mean(torch.sum(F.log_softmax(y_pred, 1)
                                        * y_true * self.weighted_vector, -1))
        else:
            # reconstruction loss without weigted vector
            BCE = -torch.mean(torch.sum(F.log_softmax(y_pred, 1) * y_true, -1))

        # calculate the 'regularization' loss based on Kullback-Leibler divergence.
        # here we compute the KLd between two multivariate normal distributions.
        # The first is a multivariate gaussian with mean 'mean' and log-variance 'log_variance'
        # The second is a multivariate standard normal distribution (mean 0 and unit variance)
        # The exact equation is given in: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(),
                                          dim=1))

        # return combined loss.
        return BCE + anneal * KLD
