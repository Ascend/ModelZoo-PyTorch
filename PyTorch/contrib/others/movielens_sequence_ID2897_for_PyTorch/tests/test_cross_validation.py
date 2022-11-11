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
import numpy as np

from spotlight import cross_validation
from spotlight.datasets import movielens


RANDOM_STATE = np.random.RandomState(42)


def test_user_based_split():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = (cross_validation
                   .user_based_train_test_split(interactions,
                                                test_percentage=0.2,
                                                random_state=RANDOM_STATE))

    assert len(train) + len(test) == len(interactions)

    users_in_test = len(np.unique(test.user_ids))
    assert np.allclose(float(users_in_test) / interactions.num_users,
                       0.2, atol=0.001)
