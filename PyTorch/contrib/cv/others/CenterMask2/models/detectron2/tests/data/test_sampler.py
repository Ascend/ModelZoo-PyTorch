# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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
import unittest
from torch.utils.data.sampler import SequentialSampler

from detectron2.data.samplers import GroupedBatchSampler


class TestGroupedBatchSampler(unittest.TestCase):
    def test_missing_group_id(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1] * 100
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual(len(mini_batch), 2)

    def test_groups(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1, 0] * 50
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual((mini_batch[0] + mini_batch[1]) % 2, 0)
