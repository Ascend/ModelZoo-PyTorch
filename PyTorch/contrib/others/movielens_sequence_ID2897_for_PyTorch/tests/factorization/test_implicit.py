# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
import pytest
import torch

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import BloomEmbedding

RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))
# Acceptable variation in specific test runs
EPSILON = .005


def test_pointwise():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='pointwise',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > 0.05


def test_bpr():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > 0.07


def test_bpr_custom_optimizer():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    def adagrad_optimizer(model_params,
                          lr=1e-2,
                          weight_decay=1e-6):

        return torch.optim.Adagrad(model_params,
                                   lr=lr,
                                   weight_decay=weight_decay)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       optimizer_func=adagrad_optimizer,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > 0.05


def test_hinge():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='hinge',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > 0.07


def test_adaptive_hinge():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='adaptive_hinge',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > 0.07


@pytest.mark.parametrize('compression_ratio, expected_mrr', [
    (0.5, 0.03),
    (1.0, 0.04),
    (1.5, 0.045),
    (2.0, 0.045),
])
def test_bpr_bloom(compression_ratio, expected_mrr):

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    user_embeddings = BloomEmbedding(interactions.num_users, 32,
                                     compression_ratio=compression_ratio,
                                     num_hash_functions=2)
    item_embeddings = BloomEmbedding(interactions.num_items, 32,
                                     compression_ratio=compression_ratio,
                                     num_hash_functions=2)
    network = BilinearNet(interactions.num_users,
                          interactions.num_items,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       representation=network,
                                       use_cuda=CUDA)

    model.fit(train)
    print(model)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr + EPSILON > expected_mrr
