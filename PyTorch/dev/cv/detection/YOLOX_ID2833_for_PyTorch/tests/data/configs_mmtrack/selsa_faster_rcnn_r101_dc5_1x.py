
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

# Copyright (c) Open-MMLab. All rights reserved.    
_base_ = [
    './faster_rcnn_r50_dc5.py', './mot_challenge.py',
    '../../../configs/_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    pretrains=None,
    detector=dict(
        backbone=dict(depth=18, base_channels=2),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=32,
                    num_attention_blocks=16)))))

# dataset settings
data = dict(
    val=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')),
    test=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# runtime settings
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
