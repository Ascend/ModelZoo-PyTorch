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

from spotlight.datasets import movielens
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel


CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


@pytest.mark.parametrize('model_class', [
    ImplicitFactorizationModel,
    ExplicitFactorizationModel
])
def test_predict_movielens(model_class):

    interactions = movielens.get_movielens_dataset('100K')

    model = model_class(n_iter=1,
                        use_cuda=CUDA)
    model.fit(interactions)

    for user_id in np.random.randint(0, interactions.num_users, size=10):
        user_ids = np.repeat(user_id, interactions.num_items)
        item_ids = np.arange(interactions.num_items)

        uid_predictions = model.predict(user_id)
        iid_predictions = model.predict(user_id, item_ids)
        pair_predictions = model.predict(user_ids, item_ids)

        assert (uid_predictions == iid_predictions).all()
        assert (uid_predictions == pair_predictions).all()
