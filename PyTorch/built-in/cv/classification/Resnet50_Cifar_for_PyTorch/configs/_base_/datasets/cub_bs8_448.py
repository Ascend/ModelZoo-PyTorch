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
# dataset settings
dataset_type = 'CUB'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=600),
    dict(type='RandomCrop', size=448),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=600),
    dict(type='CenterCrop', crop_size=448),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = 'data/CUB_200_2011/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        test_mode=True,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metric='accuracy',
    save_best='auto')  # save the checkpoint with highest accuracy
