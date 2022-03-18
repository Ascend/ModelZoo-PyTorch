# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mmpose.datasets import DATASETS


def test_mesh_Mosh_dataset():
    # test Mosh dataset
    dataset = 'MoshDataset'
    dataset_class = DATASETS.get(dataset)

    custom_dataset = dataset_class(
        ann_file='tests/data/mosh/test_mosh.npz', pipeline=[])

    _ = custom_dataset[0]


def test_mesh_H36M_dataset():
    # test H36M dataset
    dataset = 'MeshH36MDataset'
    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    _ = dataset_class(
        ann_file='tests/data/h36m/test_h36m.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]


def test_mesh_Mix_dataset():
    # test mesh Mix dataset

    dataset = 'MeshMixDataset'
    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')

    custom_dataset = dataset_class(
        configs=[
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
        ],
        partition=[0.6, 0.4])

    _ = custom_dataset[0]


def test_mesh_Adversarial_dataset():
    # test mesh Adversarial dataset

    # load train dataset
    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    train_dataset = dict(
        type='MeshMixDataset',
        configs=[
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
        ],
        partition=[0.6, 0.4])

    # load adversarial dataset
    adversarial_dataset = dict(
        type='MoshDataset',
        ann_file='tests/data/mosh/test_mosh.npz',
        pipeline=[])

    # combine train and adversarial dataset to form a new dataset
    dataset = 'MeshAdversarialDataset'
    dataset_class = DATASETS.get(dataset)
    custom_dataset = dataset_class(train_dataset, adversarial_dataset)
    item = custom_dataset[0]
    assert 'mosh_theta' in item.keys()
