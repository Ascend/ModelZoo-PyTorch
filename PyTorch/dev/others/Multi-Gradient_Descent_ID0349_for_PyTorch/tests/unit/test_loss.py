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

from loss.vae_loss import VAELoss
from loss.mse_loss import MSELoss
import pytest
import torch

# variables
y_pred = torch.tensor([1., 0.]).view(1, 2)
y_true = torch.tensor([1., 1.]).view(1, 2)
y_pred_error = torch.tensor([1, 1, 0, 0, 1]).view(1, 5)
mean = torch.tensor([0., 0.]).view(1, 2)
logvar = torch.tensor([0., 0.]).view(1, 2)
price_vector = torch.tensor([1., 2.31]).view(1, 2)
price_vector_error = torch.tensor([1., 2.31, 1.]).view(1, 3)

# vae loss for price
vae_loss_price = VAELoss(weighted_vector=price_vector)
# vae loss for relevance
vae_loss_relevance = VAELoss()
# vae loss with an explicit error
vae_loss_price_error = VAELoss(weighted_vector=price_vector_error)
# output of vae model
output_model = (y_pred, mean, logvar)
# output of vae model with an explicit error
output_model_error = (y_pred_error, mean, logvar)

# length of prediction and groundtruth are the same


def test_length_pred_gt():
    with pytest.raises(ValueError):
        vae_loss_price.compute_loss(y_pred, output_model_error)

# Given y_pred, y_true, mean and var, loss outputs expected number -> relevance_loss


def test_exact_value_relevance():
    # KL should be 0 because we compute KL divergence between two N(0,1) distributions.
    # reconstruction should be 1.62652 because of the successives functions applied to y_pred.
    assert(round(vae_loss_relevance.compute_loss(
        y_true, output_model).item(), 5) == 1.62652)

# Given y_pred, y_true, mean and var, loss outputs expected number -> price_loss


def test_exact_value_price():
    # KL should be 0 because we compute KL divergence between two N(0,1) distributions.
    # reconstruction should be 3.3469 because of the successives functions applied to y_pred.
    assert(round(vae_loss_price.compute_loss(
        y_true, output_model).item(), 5) == 3.3469)


# Mean is not None
def test_mean_not_none():
    with pytest.raises(Exception):
        vae_loss_price.compute_loss(y_true, (y_pred, None, logvar))


# variance is not None
def test_var_not_none():
    with pytest.raises(Exception):
        vae_loss_price.compute_loss(y_true, (y_pred, mean, None))


# MSELoss tests
# Incorrect arguments in compute_loss
def test_mse_loss_compute_loss_incorrect():
    mse = MSELoss()
    with pytest.raises(TypeError):
        mse.compute_loss(None, None)
    with pytest.raises(TypeError):
        mse.compute_loss(y_true, None)
    with pytest.raises(TypeError):
        mse.compute_loss(None, y_pred)
    with pytest.raises(TypeError):
        mse.compute_loss([1, 2, 3], y_pred)
    # Incorrect dimensions
    with pytest.raises(ValueError):
        mse.compute_loss(y_true, y_pred_error)


def test_mse_loss_compute_loss_correct():
    mse = MSELoss()
    assert(mse.compute_loss(y_true, y_pred) == 0.5)
    assert(mse.compute_loss(y_true, y_true) == 0)
    assert(mse.compute_loss(y_true, torch.zeros_like(y_true)) == 1)

# predictions are between 0 and 1 -> model / trainer

# ground-truth are either 0 or 1. -> dataloader
