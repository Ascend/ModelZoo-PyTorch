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

import os
import torch
import warnings
from torch import nn


"""ParetoManager class for the MAMO framework.
  The ParetoManager keeps the pareto-front updated and saves the best-model, i.e pareto optimal models.
  Typical usage example:

  foo = ParetoManager()
  for e in no_epoch:
        foo.update_pareto_front()

  More precisely, the ParetoManager will save models if they are currently optimal,
  and it will also clean them as soon as they are not optimal anymore.
"""


class ParetoManager(object):
    """ ParetoManager class that is used to update the pareto-front and save bests models,
    while removing not optimal ones.
    This class handles the update of the pareto-front,
    the saving of the best models and the removing of non dominant models for the MAMO framework.
    """

    def __init__(self, PATH='paretoFile/'):
        """Initialization of the class.

        Attributes:
        - PATH: path to store the models.
        - pareto_front : list of the pareto-front points

        - all_solutions: list of all the solutions tuples: (solution_metrics : list, solution_id: int).
                         Note: pareto optimal and dominated combined.

        - id_count: id counter to keep track the corresponding solutions with the good model update.
        """

        # Variables.
        # path to save the model.
        self.path = PATH
        # list for the pareto-front
        self._pareto_front = []
        # list of all the points
        self._all_solutions = []
        # id counter
        self.id_count = 1

    def __check_input_solution(self, solution):
        """A function that checks the input solution
        """
        if solution is None:
            raise TypeError(
                'Argument: the input solution must be a list representing a point.')
        if not isinstance(solution, list):
            raise TypeError(
                'The input solution should be a list repressenting a point!')
        if len(solution) == 0:
            raise ValueError(
                'Empty list was given as input, list should not be empty!')

    def __check_input_model(self, model):
        """A function that checks the input model
        """
        if model is None:
            raise TypeError(
                'Argument: model must be a model derived from pytorch.')
        if not isinstance(model, nn.Module):
            raise TypeError(
                'Argument: model must be a model derived from pytorch.')

    def _dominates(self, row, candidateRow):
        """A method that computes if row dominates candidateRow. It is the comparison method
        used to check if one solution dominates another, used for the computation of the add_solution method.

        Attributes:
        row: list representing a point - ex: [97, 23]
        candidateRow: list representing a point - ex: [55, 77]

        Output:
        -False if row does not dominates candidateRow
        -True if row does dominates candidateRow

        Example:
        self._dominates([2,3,7],[1,2,8]) => False
        """

        for i in range(len(row)):
            if(row[i] < candidateRow[i]):
                return False
        return True

    def _is_dominated(self, candidate_solution):
        """A method that checks if the candidate_solution is dominated or not by the solutions
        inside the current pareto front.

        Attributes:
        -candidate_solution: list representing a point - ex: [97,23]

        Output:
        -False - if candidate_solution is not dominated by any the solutions in the pareto front
        -True - if candidate_solution is dominated by a solution in the pareto front
        """
        for s, _ in self._pareto_front:
            if self._dominates(s, candidate_solution):
                return True
        return False

    def add_solution(self, solution, model):
        """A method that adds the solution to the current pareto front and maintains the
        pareto front, meaning it will remove any solution that may be dominated by adding
        the new solution.

        Attributes:
        -solution: list representing a point -ex: [97, 23]
        -model: Model which corresponds to the solution.
        """

        # Check the input solution and model
        self.__check_input_solution(solution)
        self.__check_input_model(model)

        # append an id to the solution
        current_solution = (solution, self.id_count)
        self.id_count += 1

        # append current solution to all solutions
        self._all_solutions.append(current_solution)
        # if current_solution[0] (list of points) is not dominated (not False),
        # and is not already in the pareto_front then it saves the current model
        # it also add the current_solution to the pareto_front and finally it clean the pareto_front
        # (previous solutions that are not solutions anymore are removed)
        if(current_solution[0] not in [x[0] for x in self._pareto_front]):
            if not self._is_dominated(current_solution[0]):
                self._save_model(model, current_solution)
                self._pareto_front.append(current_solution)
                self._clean_pareto_front()

    def _clean_pareto_front(self):
        """A method to clean the pareto front.
        If a new solution is added to the pareto front, it should then be updated
        and previous outdated models removed from disk.

        """

        # copy the pareto front because he will remove points from it.
        pareto_front_copy = self._pareto_front.copy()

        for current_solution in pareto_front_copy:
            self._pareto_front.remove(current_solution)

            # If the solution is still dominant, we add it to solution set
            # Else Remove/delete it from disk.
            if not self._is_dominated(current_solution[0]):
                self._pareto_front.append(current_solution)
            else:
                self._remove_model(current_solution)

    def _save_model(self, model, solution):
        """A method to save the model with a given string name.
        This method will save the model under (by default):
        '/paretoFile/<ModelName>/<ModelName>_<Epoch>'

        Where <ModelName> is the name of the pytorch derived model and <Epoch> is the epoch
        at the moment the models was stored.

        Attributes:
        - model: model that is currently trained
        """

        # retrieve the name with the coresponding identifier
        path_to_save = os.path.join(
            self.path, self._solution_to_str_rep(solution) + '.pth')

        # ignore the warning and save the model
        warnings.filterwarnings('ignore')
        torch.save(model.state_dict(), path_to_save)

    def _remove_model(self, solution):
        """A method to remove the non dominant models (models that are not in the pareto front anymore).
        """

        # retrieve the complete name
        path_to_save = os.path.join(
            self.path, self._solution_to_str_rep(solution) + '.pth')

        # check if the name is correct/exists.
        if os.path.exists(path_to_save):
            # delete it.
            os.remove(path_to_save)

    def _solution_to_str_rep(self, solution):
        """A method to format the string representation name, to save the models and also to
        be able to retrieve it, such that it can be removed later on.
        """
        s, s_id = solution
        return ('id_%s_val_metrics_') % s_id + '_'.join(['%.4f']*len(s)) % tuple(s)
