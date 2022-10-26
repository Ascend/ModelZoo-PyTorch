# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock, patch

import numpy as np

from mmcls.datasets import BaseDataset, RepeatAugSampler, build_sampler


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_single_label_dataset(length):
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    cat_ids_list = [[np.random.randint(0, 80)] for _ in range(length)]
    dataset.data_infos = MagicMock()
    dataset.data_infos.__len__.return_value = length
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    return dataset, cat_ids_list


@patch('mmcls.datasets.samplers.repeat_aug.get_dist_info', return_value=(0, 1))
def test_sampler_builder(_):
    assert build_sampler(None) is None
    dataset = construct_toy_single_label_dataset(1000)[0]
    build_sampler(dict(type='RepeatAugSampler', dataset=dataset))


@patch('mmcls.datasets.samplers.repeat_aug.get_dist_info', return_value=(0, 1))
def test_rep_aug(_):
    dataset = construct_toy_single_label_dataset(1000)[0]
    ra = RepeatAugSampler(dataset, selected_round=0, shuffle=False)
    ra.set_epoch(0)
    assert len(ra) == 1000
    ra = RepeatAugSampler(dataset)
    assert len(ra) == 768
    val = None
    for idx, content in enumerate(ra):
        if idx % 3 == 0:
            val = content
        else:
            assert val is not None
            assert content == val


@patch('mmcls.datasets.samplers.repeat_aug.get_dist_info', return_value=(0, 2))
def test_rep_aug_dist(_):
    dataset = construct_toy_single_label_dataset(1000)[0]
    ra = RepeatAugSampler(dataset, selected_round=0, shuffle=False)
    ra.set_epoch(0)
    assert len(ra) == 1000 // 2
    ra = RepeatAugSampler(dataset)
    assert len(ra) == 768 // 2
