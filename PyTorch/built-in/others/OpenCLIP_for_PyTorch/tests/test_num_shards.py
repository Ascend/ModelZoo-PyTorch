# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest

from training.data import get_dataset_size

@pytest.mark.parametrize(
    "shards,expected_size",
    [
        ('/path/to/shard.tar', 1),
        ('/path/to/shard_{000..000}.tar', 1),
        ('/path/to/shard_{000..009}.tar', 10),
        ('/path/to/shard_{000..009}_{000..009}.tar', 100),
        ('/path/to/shard.tar::/path/to/other_shard_{000..009}.tar', 11),
        ('/path/to/shard_{000..009}.tar::/path/to/other_shard_{000..009}.tar', 20),
        (['/path/to/shard.tar'], 1),
        (['/path/to/shard.tar', '/path/to/other_shard.tar'], 2),
    ]
)
def test_num_shards(shards, expected_size):
    _, size = get_dataset_size(shards)
    assert size == expected_size, f'Expected {expected_size} for {shards} but found {size} instead.'
