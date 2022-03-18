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

from metric.recall_at_k import RecallAtK
from loss.vae_loss import VAELoss
from dataloader.mamo_dataset import MamoDataset
from validator import Validator
from torch.utils.data import DataLoader
from models.multi_VAE import MultiVAE
import os
import numpy as np
import pytest
import yaml

# Packages needed to run test:
# os
# numpy
# torch
# pytest
# yaml
with open('yaml_files/data_info.yaml', 'r') as stream:
    try:
        data_info = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# create temporary directories
if not os.path.isdir('test_data_val'):
    os.mkdir('test_data_val')

# generate random data
np.random.seed(42)
dir_path = 'test_data_val/'

test_input_data_path = os.path.join(
    dir_path, 'movielens_small_test_input.npy')
test_output_data_path = os.path.join(
    dir_path, 'movielens_small_test_test.npy')

np.save(test_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_output_data_path, np.random.rand(2000, 8936).astype('float32'))

# Variables
dataset = MamoDataset(np.load(test_input_data_path),
                      np.load(test_output_data_path))

model = MultiVAE(params='yaml_files/params_multi_VAE.yaml')
model.initialize_model()
dataloader = DataLoader(dataset, batch_size=data_info['batch_size'],
                        shuffle=True, drop_last=True)
metrics = [RecallAtK(10)]
objectives = [VAELoss()]

obj_results = [0.4, 0.5, 0.7]
alphas = [0.5, 0.2, 0.3]
max_normalization = [1, 0.5, 2]


# A Validator object cannot be created without a model.
def test_validator_init_no_model():
    with pytest.raises(TypeError, match='Argument: model must be set.'):
        Validator(None, dataloader, metrics, objectives)


# A Validator object cannot be created without a dataloader.
def test_validator_init_no_dataloader():
    with pytest.raises(TypeError, match='Argument: dataloader must be set.'):
        Validator(model, None, metrics, objectives)


# A Validator object cannot be created without metrics and objectives.
def test_validator_init_no_metrics_and_objectives():
    with pytest.raises(TypeError, match='Either argument: metrics or argument:'
                       + ' objectives must be set.'):
        Validator(model, dataloader, None, None)


# A Validator object cannot be created with an incorrect model.
def test_validator_init_bad_model():
    with pytest.raises(TypeError, match='Argument: model must be derived'
                                        + ' from nn.Module.'):
        Validator('model', dataloader, metrics, objectives)


# A Validator object cannot be created with an incorrect dataloader.
def test_validator_init_bad_dataloader():
    with pytest.raises(TypeError, match='Argument: dataloader must be a'
                                        + ' pytorch DataLoader.'):
        Validator(model, 'dataloader', metrics, objectives)


# A Validator object cannot be created with an incorrect metrics argument.
def test_validator_init_bad_metrics():
    with pytest.raises(TypeError, match='Argument: metrics must be a list.'):
        Validator(model, dataloader, 'metrics', objectives)
    with pytest.raises(TypeError, match='All elements of argument: metrics'
                                        + ' must be of type MetricAtK.'):
        Validator(model, dataloader, ['metric1', RecallAtK(2)], objectives)


# A Validator object cannot be created with an incorrect objectives argument.
def test_validator_init_bad_objectives():
    with pytest.raises(TypeError, match='Argument: objectives must'
                                        + ' be a list.'):
        Validator(model, dataloader, metrics, 'objectives')
    with pytest.raises(TypeError, match='All elements of argument: objectives'
                                        + ' must be of type Loss.'):
        Validator(model, dataloader, metrics, ['objective', VAELoss()])


# Testing the combine_objectives method
# combine_objectives cannot run if missing obj_results or in incorrect format
def test_validator_combine_objectives_bad_obj_results():
    v = Validator(model, dataloader, metrics, objectives)
    with pytest.raises(TypeError, match='Argument: obj_results must be set.'):
        v.combine_objectives(None, alphas, max_normalization)
    with pytest.raises(TypeError, match='Argument:'
                       + ' obj_results must be a list.'):
        v.combine_objectives('Results', alphas, max_normalization)
    with pytest.raises(TypeError, match='All elements of argument: obj_results'
                       + ' must be of type int or float.'):
        v.combine_objectives([1, 2.5, 'number'], alphas, max_normalization)


# combine_objectives cannot run if alphas is in incorrect format
def test_validator_combine_objectives_bad_alphas():
    v = Validator(model, dataloader, metrics, objectives)
    with pytest.raises(TypeError, match='Argument:'
                       + ' alphas must be a list.'):
        v.combine_objectives(obj_results, 'alphas', max_normalization)
    with pytest.raises(TypeError, match='All elements of argument: alphas'
                       + ' must be of type int or float.'):
        v.combine_objectives(obj_results, [1, 2.5, 'number'],
                             max_normalization)
    with pytest.raises(ValueError, match='The length of alphas must be equal'
                       + ' to that of obj_results'):
        v.combine_objectives(obj_results, [1, 2.5], max_normalization)


# combine_objectives cannot run if max_normalization is in incorrect format
def test_validator_combine_objectives_bad_max_normalization():
    v = Validator(model, dataloader, metrics, objectives)
    with pytest.raises(TypeError, match='Argument:'
                       + ' max_normalization must be a list.'):
        v.combine_objectives(obj_results, alphas, 'max_normalization')
    with pytest.raises(TypeError, match='All elements of argument:'
                       + ' max_normalization must be of type int or float.'):
        v.combine_objectives(obj_results, alphas, [1, 2.5, 'number'])
    with pytest.raises(ValueError, match='The length of max_normalization must'
                       + ' be equal to that of obj_results'):
        v.combine_objectives(obj_results, alphas, [1, 2.5])


# Correct runs of combine_objectives
def test_validator_combine_objectives_no_problem():
    v = Validator(model, dataloader, metrics, objectives)
    assert(v.combine_objectives(obj_results, alphas, max_normalization)
           == 0.505)
    assert(v.combine_objectives(obj_results, None, max_normalization) == 1.75)
    assert(v.combine_objectives(obj_results, alphas, None) == 0.51)
    assert(v.combine_objectives(obj_results) == sum(obj_results))


def test_validator_evaluate_bad_inputs():
    v = Validator(model, dataloader, metrics, objectives)
    with pytest.raises(TypeError, match='Argument: disable_anneal'
                       + ' must be a bool.'):
        v.evaluate(disable_anneal='True')
    with pytest.raises(TypeError, match='Argument: verbose must be a bool.'):
        v.evaluate(verbose='True')


# Small test just to show it works
def test_validator_evaluate_no_problem():
    v = Validator(model, dataloader, metrics, objectives)
    results = v.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert isinstance(results[1], list)


# removing generated data
def test_cleanup():
    os.remove(test_input_data_path)
    os.remove(test_output_data_path)
    os.rmdir('test_data_val')
