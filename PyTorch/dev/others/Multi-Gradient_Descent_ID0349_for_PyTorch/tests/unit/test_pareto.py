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

from paretomanager.pareto_manager_class import ParetoManager
import pytest
from torch import nn
import os
import shutil

# variables

scores = [[13, 2, 3], [4, 5, 6], [7, 45, 9], [10, 11, 12], [
    12, 44, 11], [13, 2, 3]]
# should be [[13,2,3], [7,45,9], [10,11,12], [12, 44, 11]]
# without saving the last [13,2,3] because it's already there.


path = 'paretoFile/'
os.mkdir(path)

# dummy class


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return x + 1


manager_test = ParetoManager()
model_test = MyModel()

# Check that input is not an empty list


def test_input_is_not_empty_list():
    with pytest.raises(ValueError, match='Empty list was given as input, list should not be empty!'):
        manager_test.add_solution([], model_test)

# check that input is a list


def test_input_is_list():
    with pytest.raises(TypeError, match='The input solution should be a list repressenting a point!'):
        manager_test.add_solution((23, 12), model_test)

# check that model is derived from pytorch.


def test_model_is_from_pytorch():
    with pytest.raises(TypeError, match='Argument: model must be a model derived from pytorch.'):
        manager_test.add_solution([12, 12], 'model')

# check that input is not None.


def test_input_not_none():
    with pytest.raises(TypeError, match='Argument: the input solution must be a list representing a point.'):
        manager_test.add_solution(None, model_test)

# check model is not None.


def test_model_not_none():
    with pytest.raises(TypeError, match='Argument: model must be a model derived from pytorch.'):
        manager_test.add_solution([12, 12], None)

# Check that the output is expected for scores


def test_exact_value_scores():
    # manager
    manager = ParetoManager()
    # dummy class
    model = MyModel()

    for i in range(len(scores)):
        manager.add_solution(scores[i], model)

    # check that both the founded solution and the expected one are equals.
    set_a = set([tuple(y[0]) for y in manager._pareto_front])
    set_b = set([tuple(y)
                 for y in [[10, 11, 12], [13, 2, 3], [7, 45, 9], [12, 44, 11]]])

    assert(
        len(set_a - set_b) == 0
    )


def _solution_to_str_rep(solution):
    s, s_id = solution
    return ('id_%s_val_metrics_') % s_id + '_'.join(['%.4f']*len(s)) % tuple(s)

# Check it deletes correctly


def test_deletes_correctly_scores():
    # manager
    manager = ParetoManager()
    # dummy class
    model = MyModel()

    for i in range(len(scores)):
        manager.add_solution(scores[i], model)

    # The pareto optimal points should be [[13,2,3], [7,45,9], [10,11,12], [12, 44, 11]] added respectively
    # in epochs 1, 3, 4 and 5. So the remaining saved models should be the ones after epoch 1, 3, 4 and 5.
    # Model after epoch 2 should be deleted.
    path1 = os.path.join(path, _solution_to_str_rep((scores[0], 1)) + '.pth')
    path2 = os.path.join(path, _solution_to_str_rep((scores[1], 2)) + '.pth')
    path3 = os.path.join(path, _solution_to_str_rep((scores[2], 3)) + '.pth')
    path4 = os.path.join(path, _solution_to_str_rep((scores[3], 4)) + '.pth')
    path5 = os.path.join(path, _solution_to_str_rep((scores[4], 5)) + '.pth')
    path6 = os.path.join(path, _solution_to_str_rep((scores[5], 5)) + '.pth')

    keep1 = os.path.exists(path1)
    keep2 = os.path.exists(path2)
    keep3 = os.path.exists(path3)
    keep4 = os.path.exists(path4)
    keep5 = os.path.exists(path5)
    keep6 = os.path.exists(path6)

    keep = (keep1 and keep3 and keep4 and keep5)
    remove = (keep2 and keep6)

    assert((keep is True) and (remove is False))


# removing generated dir
def test_cleanup():
    shutil.rmtree(path)
