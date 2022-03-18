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

Validator class, used for validating models on datasets with specified
 metrics and/or objectives.

The Validator class is used during the validation stage of training a new model
and for testing the performance of a pretrained model on a novel dataset.
"""
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from metric.metric_at_k import MetricAtK
from loss.loss_class import Loss
from models.multi_VAE import MultiVAE


class Validator:
    """Validator class.

    The Validator class has two uses. The first is during the training of a
    novel model, when the Validator is used for validation, and the second
    is for the testing of a pretrained model on a new dataset, in order to
    judge its performance. The Validator must have at least metrics or
    objectives, it cannot have neither.

    Attributes:
        _model: A pytorch model derived from nn.Module the performance of which
            is being assesed by the Validator.
        _dataloader: A pytorch DataLoader for obtaining the input data.
        _metrics: An optional list of MetricAtK objects.
        _objectives: An optional list of Loss objects.
    """
    def __init__(self, model, dataloader, metrics, objectives):
        """Inits the Validator with the necassary arguments. Either metrics or
        objectives must be passed.

        Args:
            model: A pytorch model derived from nn.Module the performance of
            which is being assesed by the Validator.
            dataloader: A pytorch DataLoader for obtaining the input data.
            metrics: An optional list of MetricAtK objects.
            objectives: An optional list of Loss objects.
        Raises:
            TypeError: If any of the arguments are not set or if both metrics
                and objectives are not set. Also raises this error if any of
                the arguments are of the incorrect type.
        """
        # Missing argument errors
        if model is None:
            raise TypeError('Argument: model must be set.')
        elif dataloader is None:
            raise TypeError('Argument: dataloader must be set.')
        elif metrics is None and objectives is None:
            raise TypeError('Either argument: metrics or argument: objectives'
                            + ' must be set.')
        # Wrong type argument errors
        if not isinstance(model, nn.Module):
            raise TypeError('Argument: model must be derived from'
                            + ' nn.Module.')
        elif not isinstance(dataloader, DataLoader):
            raise TypeError('Argument: dataloader must be a pytorch'
                            + ' DataLoader.')
        elif metrics and not isinstance(metrics, list):
            raise TypeError('Argument: metrics must be a list.')
        elif metrics and not all(isinstance(m, MetricAtK) for m in metrics):
            raise TypeError('All elements of argument: metrics must be'
                            + ' of type MetricAtK.')
        elif objectives and not isinstance(objectives, list):
            raise TypeError('Argument: objectives must be a list.')
        elif objectives and not all(isinstance(o, Loss) for o in objectives):
            raise TypeError('All elements of argument: objectives must be'
                            + ' of type Loss.')

        self._model = model
        self._dataloader = dataloader
        self._metrics = metrics
        self._objectives = objectives
        self._stats_metrics = np.array([])
        self._stats_objectives = np.array([])

    def evaluate(self, disable_anneal=False, verbose=False):
        """A method that runs the validation of the model on the dataset.

        Evaluate the performance of the model on the passed dataset using the
        metrics and objectives in the Validator object.

        Args:
            verbose: A bool value determining whether we print to stdout.
                default = True.
            disable_anneal: A bool signifying whether to use 0 annealing value
                in some loss functions, default=False.

        Returns:
            A tuple consiting of a list of results of the metric evaluation and
            a list of results of the objective evaluation
        Raises:
            TypeError: If disable_anneal or verbose are not of type bool.
        """
        if not isinstance(disable_anneal, bool):
            raise TypeError('Argument: disable_anneal must be a bool.')
        if not isinstance(verbose, bool):
            raise TypeError('Argument: verbose must be a bool.')
        # Set npu if available
        device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        self._model.to(device)
        self._model.eval()
        # Initialise statistics if applicable
        if self._metrics:
            self._stats_metrics = np.zeros_like(self._metrics, dtype=float)
        if self._objectives:
            self._stats_objectives = np.zeros_like(self._objectives, dtype=float)
        cnt = 0
        with torch.no_grad():
            for batch in self._dataloader:
                x, y = batch
                x = Variable(x).to(device)
                y = Variable(y).to(device)
                # Some models may return more than just their predictions.
                # For these models the first element of the returned tuple
                # should contain the predictions. These models are:
                # MultiVAE,  (add to list as it grows, and modify code)
                model_output = self._model(x)
                if isinstance(self._model, MultiVAE)\
                   and isinstance(model_output, tuple):
                    # Filtering out already chosen items:
                    # In order to make sure we are not recommending already
                    # chosen items, here we set the prediction of the already
                    # chosen items to the minimum value. We know what items
                    # have already been chosen since they are in our input X.
                    model_output[0][x == 1] = torch.min(model_output[0])
                    y_out = model_output[0]
                else:
                    y_out = model_output
                    # Filtering out already chosen elements
                    y_out[x == 1] = torch.min(y_out)
                cnt += 1
                # Compute moving average with the new batch for each metric
                if self._metrics:
                    self._stats_metrics = self._stats_metrics\
                     * ((cnt - 1)/cnt)\
                     + np.array([metric.evaluate(y, y_out)
                                 for metric in self._metrics]) * (1/cnt)
                # Compute moving average with the new batch for each objective
                if self._objectives:
                    if disable_anneal:
                        self._stats_objectives = self._stats_objectives\
                         * ((cnt - 1)/cnt)\
                         + np.array([objective.compute_loss(y, model_output,
                                                            0).item()
                                    for objective in self._objectives])\
                         * (1/cnt)
                    else:
                        self._stats_objectives = self._stats_objectives\
                         * ((cnt - 1)/cnt)\
                         + np.array([objective.compute_loss(y,
                                    model_output).item()
                                    for objective in self._objectives])\
                         * (1/cnt)
        # Logging and printing of results
        if self._objectives:
            log_str = 'Validation loss:' + ''.join([' {}: {:.2f}'.format(
              self._objectives[i].name, self._stats_objectives[i])
              for i in range(len(self._objectives))])
            logging.info(log_str)
            if verbose:
                print('='*90)
                print(log_str)
        if self._metrics:
            log_str = 'Validation metrics:' + ''.join([' {}: {:.2f}'.format(
              self._metrics[i].get_name(), self._stats_metrics[i])
              for i in range(len(self._metrics))])
            logging.info(log_str)
            if verbose:
                print('='*90)
                print(log_str)
        return self._stats_metrics.tolist(), self._stats_objectives.tolist()

    def combine_objectives(self, obj_results, alphas=None,
                           max_normalization=None):
        """A method combines the values passed to it.

        Combine the results of objectives/losses passed using alphas and
        max normalization, if set. Used after validation by the Trainer.
        Example:
            results = validator.evaluate()
            validation_loss = validator.combine_objectives(results[1],
                                                           alphas,
                                                           max_normalization)

        Args:
            obj_results: A list of floats.
            alphas: A list of alpha values to be multiplied with the
                objectives, default = None
            max_normalization: A list of values to divide the objectives with,
                default = None

        Returns:
            A float containing the combined value of the objectives.
        Raises:
            TypeError: If obj_results is not set. Also raises this error if any
                of the arguments are of the incorrect type.
            ValueError: If obj_results, alphas and max_normalization do not
            have the same dimensions.
        """
        # Missing argument error
        if obj_results is None:
            raise TypeError('Argument: obj_results must be set.')
        # Wrong type argument errors
        if not isinstance(obj_results, list):
            raise TypeError('Argument: obj_results must be a list.')
        elif alphas and not isinstance(alphas, list):
            raise TypeError('Argument: alphas must be a list.')
        elif max_normalization and not isinstance(max_normalization, list):
            raise TypeError('Argument: max_normalization must be a list.')

        if not all(isinstance(i, (np.floating, float, np.integer, int))
                   for i in obj_results):
            raise TypeError('All elements of argument: obj_results must be'
                            + ' of type int or float.')
        #elif alphas and not all(isinstance(i, (np.floating, float,
        #                                       np.integer, int))
        #                        for i in alphas):
        #    raise TypeError('All elements of argument: alphas must be'
        #                    + ' of type int or float.')
        elif max_normalization and not all(isinstance(i, (np.floating, float,
                                                          np.integer, int))
                                           for i in max_normalization):
            raise TypeError('All elements of argument: max_normalization must'
                            + ' be of type int or float.')
        # Incorrect dimensions
        if alphas and len(alphas) != len(obj_results):
            raise ValueError('The length of alphas must be equal to'
                             + ' that of obj_results')
        if max_normalization and len(max_normalization) != len(obj_results):
            raise ValueError('The length of max_normalization must be equal to'
                             + ' that of obj_results')

        if alphas is None:
            alphas = [1]*len(obj_results)
        if max_normalization is None:
            max_normalization = [1]*len(obj_results)
        return sum(1/max_normalization[i]*alphas[i]*obj_results[i]
                   for i in range(len(alphas)))
