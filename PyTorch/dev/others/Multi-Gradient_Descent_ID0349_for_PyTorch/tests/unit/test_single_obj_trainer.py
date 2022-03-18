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
"""
Copyright 漏 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import torch
import numpy as np
import os
import pytest

from dataloader.ae_data_handler import AEDataHandler
from models.multi_VAE import MultiVAE
from loss.vae_loss import VAELoss
from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from paretomanager.pareto_manager_class import ParetoManager
from validator import Validator
from single_objective_trainer import SingleObjectiveTrainer
import torch.nn as nn
from torch.utils.data import DataLoader

# set npu if available
device = torch.device('npu' if torch.npu.is_available() else 'cpu')

# create temporary directories
if not os.path.isdir('test_data_so'):
    os.mkdir('test_data_so')
if not os.path.isdir('test_data_so/models'):
    os.mkdir('test_data_so/models')

# generate random data
np.random.seed(42)
dir_path = 'test_data_so/'
train_data_path = os.path.join(
    dir_path, 'movielens_small_training.npy')
validation_input_data_path = os.path.join(
    dir_path, 'movielens_small_validation_input.npy')
validation_output_data_path = os.path.join(
    dir_path, 'movielens_small_validation_test.npy')
test_input_data_path = os.path.join(
    dir_path, 'movielens_small_test_input.npy')
test_output_data_path = os.path.join(
    dir_path, 'movielens_small_test_test.npy')
products_data_path = os.path.join(
    dir_path, 'movielens_products_data.npy')


np.save(train_data_path, np.random.rand(10000, 8936).astype('float32'))
np.save(validation_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(validation_output_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_output_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(products_data_path, np.random.rand(8936))

dataHandler = AEDataHandler(
    'Testing trainer random dataset', train_data_path, validation_input_data_path,
    validation_output_data_path, test_input_data_path,
    test_output_data_path)

input_dim = dataHandler.get_input_dim()
output_dim = dataHandler.get_output_dim()

products_data_np = np.load(products_data_path)
products_data_torch = torch.tensor(
    products_data_np, dtype=torch.float32).to(device)

# create model
model = MultiVAE(params='yaml_files/params_multi_VAE.yaml')

correctness_loss = VAELoss()
revenue_loss = VAELoss(weighted_vector=products_data_torch)
losses = [correctness_loss, revenue_loss]

recallAtK = RecallAtK(k=10)
revenueAtK = RevenueAtK(k=10, revenue=products_data_np)
validation_metrics = [recallAtK, revenueAtK]

# Set up this
save_to_path = 'test_data_so/models'
yaml_path = 'yaml_files/trainer_params.yaml'


# test the init arguments
def test_check_input1():
    with pytest.raises(TypeError, match='Please check you are using the right data handler object,'
                       + ' or the right order of the attributes!'):
        trainer = SingleObjectiveTrainer(None, model, correctness_loss,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()
    with pytest.raises(TypeError, match='Please check you are using the right data handler object,'
                       + ' or the right order of the attributes!'):
        trainer = SingleObjectiveTrainer(model, dataHandler, correctness_loss,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input2():
    with pytest.raises(TypeError, match='Please check you are using the right model object,'
                       + ' or the right order of the attributes!'):
        trainer = SingleObjectiveTrainer(dataHandler, None, correctness_loss,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input3():
    class TestModel(nn.Module):
        def forward(self):
            return 1
    with pytest.raises(TypeError, match='Please check if your models has initialize_model\\(\\) method defined!'):
        trainer = SingleObjectiveTrainer(dataHandler, TestModel(), correctness_loss,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input4():
    with pytest.raises(TypeError, match='Please check you are using the right loss object,'
                       + ' or the right order of the attributes!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, losses,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()
    with pytest.raises(TypeError, match='Please check you are using the right loss object,'
                       + ' or the right order of the attributes!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, None,
                                         validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input5():
    with pytest.raises(TypeError, match='Please check you are using the right metric objects,'
                       + ' or the right order of the attributes!'):
        validation_metrics_tmp = validation_metrics.copy()
        validation_metrics_tmp[0] = model
        trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                         validation_metrics_tmp, save_to_path, yaml_path)
        trainer.train()


def test_check_input6():
    with pytest.raises(ValueError, match='Please make sure that the directory'
                       + ' where you want to save the models is empty!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                         validation_metrics, '.', yaml_path)
        trainer.train()


def test_check_input8():
    # check for None metrics
    with pytest.raises(ValueError, match='The validation_metrics are None,'
                       + ' please make sure to give valid validation_metrics!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                         None, save_to_path, yaml_path)
        trainer.train()
    # check if length is at least 1
    validation_metrics_tmp = []
    with pytest.raises(ValueError, match='Please check you have defined at least one validation metric!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                         validation_metrics_tmp, save_to_path, yaml_path)
        trainer.train()


def test_check_input9():
    with pytest.raises(TypeError, match='Please make sure that the optimizer is a pytorch Optimizer object!'):
        trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                         validation_metrics, save_to_path, yaml_path, model)
        trainer.train()


# test the reading from the yaml files
def test_read_yaml_params():
    trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                     validation_metrics, save_to_path, yaml_path)
    assert trainer.seed == 42
    assert trainer.learning_rate == 1e-3
    assert trainer.batch_size_training == 500
    assert trainer.shuffle_training is True
    assert trainer.drop_last_batch_training is True
    assert trainer.batch_size_validation == 500
    assert trainer.shuffle_validation is True
    assert trainer.drop_last_batch_validation is False
    assert trainer.number_of_epochs == 50
    assert trainer.anneal is True
    assert trainer.beta_start == 0
    assert trainer.beta_cap == 0.3
    assert trainer.beta_step == 0.3/10000


# test the init of the objects
def test_init_objects():
    trainer = SingleObjectiveTrainer(dataHandler, model, correctness_loss,
                                     validation_metrics, save_to_path, yaml_path)
    assert type(trainer._train_dataloader) == DataLoader
    assert type(trainer.pareto_manager) == ParetoManager
    assert trainer.pareto_manager.path == save_to_path
    assert type(trainer.validator) == Validator


# removing generated data
def test_cleanup():
    os.remove(train_data_path)
    os.remove(validation_input_data_path)
    os.remove(validation_output_data_path)
    os.remove(test_input_data_path)
    os.remove(test_output_data_path)
    os.remove(products_data_path)
    os.rmdir('test_data_so/models')
    os.rmdir('test_data_so')
