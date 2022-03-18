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

# COPSolver unit test

from copsolver.analytical_solver import AnalyticalSolver
from copsolver.frank_wolfe_solver import FrankWolfeSolver
import numpy as np
import pytest

# variables
gradient = np.zeros(shape=(2, 5))
gradient[0] = np.array([0, 1, 0.5, 0.5, -2])
gradient[1] = np.array([1, 0, 0, -0.5, 0.5])

gradient_1 = np.zeros(shape=(2, 6))
gradient_1[0] = np.array([1, 2, 3, 4, 5, 6])
gradient_1[1] = np.array([1, 1, 6, 4, 3, 2])

gradient_false = np.array([[0, 1, 0.5], [0.5, -2]], dtype=object)
gradient_same = np.array([[0, 1, 0.5], [0, 1, 0.5]], dtype=object)

analytical = AnalyticalSolver()
frank_wolfe = FrankWolfeSolver(max_iter=100, min_change=1e-3)


# analytical solver cannot take none as an argument
def test_analytical_none_argument():
    with pytest.raises(TypeError):
        analytical.solve(None)


# analytical solver cannot take less than 2 gradients
def test_analytical_less_than_2_gradients():
    with pytest.raises(ValueError):
        analytical.solve(np.ones(1))


# analytical solver cannot take more than 2 gradients
def test_analytical_more_than_2_gradients():
    with pytest.raises(ValueError):
        analytical.solve(np.ones(3))


# analytical sovler cannot take gradients of different size
def test_analytical_differents_gradient_length():
    with pytest.raises(ValueError):
        analytical.solve(gradient_false)


# analytical sovler cannot take the same gradients
def test_analytical_same_gradient():
    assert((analytical.solve(gradient_same) == [0.5,0.5]))


# analytical solver gives correct alphas
def test_analytical_correctness():
    assert((analytical.solve(gradient) == [0.28947368421052633, 1-0.28947368421052633]))
    assert((analytical.solve(gradient_1) == [0.1, 0.9]))


# Frank Wolfe and Analytical Solver returns same alpha for 2 gradients
def test_analytical_and_FW_2_gradients():
    assert((analytical.solve(gradient) == frank_wolfe.solve(gradient)).all())
    assert((analytical.solve(gradient_1) == frank_wolfe.solve(gradient_1)).all())


# Frank Wolfe raises an error is max_iter < 0
def test_FW_max_iter_less_than_0():
    with pytest.raises(ValueError):
        FrankWolfeSolver(max_iter=-1)


# Frank Wolfe raises an error is min_change < 0
def test_FW_min_change_less_than_0():
    with pytest.raises(ValueError):
        FrankWolfeSolver(min_change=-1)


# Frank Wolfe raises an error if gradients have different lengths
def test_FE_different_gradients_length():
    with pytest.raises(ValueError):
        frank_wolfe.solve(gradient_false)
