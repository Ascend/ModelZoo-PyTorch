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


def test_bottom_up_COCO_dataset():
    dataset = 'BottomUpCocoDataset'
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        dataset_joints=17,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        ])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128, 256],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=2,
        scale_aware_sigma=False)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]


def test_bottom_up_CrowdPose_dataset():
    dataset = 'BottomUpCrowdPoseDataset'
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=14,
        dataset_joints=14,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128, 256],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=2,
        scale_aware_sigma=False)

    _ = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    image_id = 103319
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]
