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
import torch
import numpy as np
from scipy.special import softmax
from py_tabnet.utils import PredictDataset, filter_weights
from py_tabnet.abstract_model import TabModel
from py_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class TabNetClassifier(TabModel):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        #return self.loss_fn(y_pred, y_true.long())
        return self.loss_fn(y_pred, y_true.to(torch.int32))
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(f'npu:{NPU_CALCULATE_DEVICE}').float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
