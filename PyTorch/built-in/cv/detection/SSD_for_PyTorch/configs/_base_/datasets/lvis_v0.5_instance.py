# Copyright 2022 Huawei Technologies Co., Ltd.
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
# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV05Dataset'
data_root = 'data/lvis_v0.5/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v0.5_train.json',
            img_prefix=data_root + 'train2017/')),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/'))
evaluation = dict(metric=['bbox', 'segm'])
