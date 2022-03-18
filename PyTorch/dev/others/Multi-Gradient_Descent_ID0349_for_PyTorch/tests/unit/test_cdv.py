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

Unit Tests for CommonDescentVector
"""

import pytest
import torch
import numpy as np
from torch.autograd import Variable
from copsolver.analytical_solver import AnalyticalSolver
from commondescentvector.single_objective_cdv import SingleObjectiveCDV
from commondescentvector.multi_objective_cdv import MultiObjectiveCDV

# variables

analytical = AnalyticalSolver()


loss_1 = Variable(torch.tensor([4.]), requires_grad=True)
max_empirical_loss_1 = Variable(torch.tensor([2.]))
loss_2 = Variable(torch.tensor([16.]), requires_grad=True)
max_empirical_loss_2 = Variable(torch.tensor([4.]))
max_empirical_loss_neg = Variable(torch.tensor([0.]))

single_cdv = SingleObjectiveCDV()
single_cdv_normalized = SingleObjectiveCDV(max_empirical_loss_1, True)

gradient = np.zeros(shape=(2, 6))
gradient[0] = np.array([1, 2, 3, 4, 5, 6])
gradient[1] = np.array([1, 1, 6, 4, 3, 2])

losses = np.array([loss_1, loss_2])
max_empirical_losses = np.array([max_empirical_loss_1, max_empirical_loss_2])

multi_cdv = MultiObjectiveCDV(analytical)
multi_cdv_normalized = MultiObjectiveCDV(analytical, max_empirical_losses, True)

alpha_base = analytical.solve(gradient)
gradient_norm = gradient.copy()
for i in range(len(gradient)):
    gradient_norm[i] /= max_empirical_losses[i]
alpha_norm_base = analytical.solve(gradient_norm)


# ==========  SingleObjective ==========


# Single objective get_descent_vector gives correct loss
def test_single_objective_correctness():
    assert (single_cdv.get_descent_vector([loss_1])[0].data == loss_1.data)


# Single objective get_descent_vector gives correct normalized loss
def test_single_objective_normalized_correctness():
    assert (single_cdv_normalized.get_descent_vector([loss_1]).data == max_empirical_loss_1.data)


# Setting the max_empirical_loss works
def test_single_objective_set_max_empirical():
    single_cdv_tmp = SingleObjectiveCDV(normalized=True)
    single_cdv_tmp.set_max_empirical_loss(max_empirical_loss_1)
    assert (single_cdv_tmp.get_descent_vector([loss_1]).data == max_empirical_loss_1.data)


# Setting the normalized works
def test_single_objective_set_normalized():
    single_cdv_tmp = SingleObjectiveCDV(max_empirical_loss=max_empirical_loss_1)
    single_cdv_tmp.set_normalized(True)
    assert (single_cdv_tmp.get_descent_vector([loss_1]).data == max_empirical_loss_1.data)


# Losses cannot be none in get_descent_vector
def test_single_objective_get_vector_none():
    with pytest.raises(TypeError):
        single_cdv.get_descent_vector(None)


# Cannot pass more than 1 loss in get_descent_vector
def test_single_objective_more_than_one_loss():
    with pytest.raises(ValueError):
        single_cdv.get_descent_vector([loss_1, loss_1])


# Cannot normalize if max_empirical_loss is not set
def test_single_objective_max_loss_not_set():
    with pytest.raises(TypeError):
        single_cdv_tmp = SingleObjectiveCDV(normalized=True)
        single_cdv_tmp.get_descent_vector([loss_1])


# max_empirical_loss should be greater than zero
def test_single_objective_max_loss_negative():
    with pytest.raises(ValueError):
        SingleObjectiveCDV(max_empirical_loss=max_empirical_loss_neg)


def test_single_objective_set_max_loss_negative():
    with pytest.raises(ValueError):
        single_cdv.set_max_empirical_loss(max_empirical_loss_neg)


# ==========  MultiObjective ==========

# Multi-objective get_descent_vector gives correct loss
def test_multi_objective_correctness():
    final_loss, alphas = multi_cdv.get_descent_vector(losses, gradient)
    assert(final_loss.data == ((alphas[0]*loss_1) + (alphas[1]*loss_2)).data)
    assert(alphas == alpha_base)


# Multi-objective normalized get_descent_vector gives correct loss
def test_multi_objective_normalized_correctness():
    final_loss, alphas = multi_cdv_normalized.get_descent_vector(losses, gradient)
    assert(final_loss.data == ((alphas[0]*max_empirical_loss_1) + (alphas[1]*max_empirical_loss_2)).data)
    assert(alphas == alpha_norm_base)


# Setting the max_empirical_loss works
def test_multi_objective_set_max_empirical():
    multi_cdv_tmp = MultiObjectiveCDV(analytical, normalized=True)
    multi_cdv_tmp.set_max_empirical_losses(max_empirical_losses)
    final_loss, alphas = multi_cdv_tmp.get_descent_vector(losses, gradient)
    assert(final_loss.data == ((alphas[0]*max_empirical_loss_1) + (alphas[1]*max_empirical_loss_2)).data)
    assert(alphas == alpha_norm_base)


# Setting the normalized works
def test_multi_objective_set_normalized():
    multi_cdv_tmp = MultiObjectiveCDV(analytical, max_empirical_losses=max_empirical_losses)
    multi_cdv_tmp.set_normalized(True)
    final_loss, alphas = multi_cdv_tmp.get_descent_vector(losses, gradient)
    assert(final_loss.data == ((alphas[0]*max_empirical_loss_1) + (alphas[1]*max_empirical_loss_2)).data)
    assert(alphas == alpha_norm_base)


# Cannot instantiate with None copsolver
def test_multi_objective_none_copsolver():
    with pytest.raises(TypeError):
        MultiObjectiveCDV(None)


# Losses cannot be None
def test_multi_objective_none_losses():
    with pytest.raises(TypeError):
        multi_cdv.get_descent_vector(None, gradient)


# Gradient cannot be None
def test_multi_objective_none_gradients():
    with pytest.raises(TypeError):
        multi_cdv.get_descent_vector(losses, None)


# Losses cannot be empty
def test_multi_objective_empty_losses():
    with pytest.raises(ValueError):
        multi_cdv.get_descent_vector([], gradient)


# Gradients cannot be empty
def test_multi_objective_empty_gradients():
    with pytest.raises(ValueError):
        multi_cdv.get_descent_vector(losses, [])


# Losses and gradients must be the same size
def test_multi_objective_gradients_losses_same():
    with pytest.raises(ValueError):
        multi_cdv.get_descent_vector(losses, np.zeros(shape=(3, 1)))


# Cannot normalize if max_empirical_loss is not set
def test_multi_objective_max_loss_not_set():
    with pytest.raises(TypeError):
        multi_cdv_tmp = MultiObjectiveCDV(analytical, normalized=True)
        multi_cdv_tmp.get_descent_vector(losses, gradient)


# Losses must be the same size as max_empirical_losses
def test_multi_objective_losses_size():
    with pytest.raises(ValueError):
        multi_cdv_normalized.get_descent_vector([loss_1], gradient)


# gradients must be the same size as max_empirical_losses
def test_multi_objective_gradient_size():
    with pytest.raises(ValueError):
        multi_cdv_normalized.get_descent_vector(losses, np.zeros(shape=(1, 1)))


# max_empirical_loss should be greater than zero
def test_multi_objective_max_loss_negative():
    with pytest.raises(ValueError):
        MultiObjectiveCDV(analytical, max_empirical_losses=[max_empirical_loss_neg, max_empirical_loss_neg])


def test_multi_objective_set_max_loss_negative():
    with pytest.raises(ValueError):
        multi_cdv.set_max_empirical_losses([max_empirical_loss_neg, max_empirical_loss_neg])
