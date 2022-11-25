# Copyright (c) Facebook, Inc. and its affiliates.
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
# --------------------------------------------------------
_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400)
checkpoint_config = dict(by_epoch=False, interval=32000)
evaluation = dict(interval=32000, metric='mIoU')

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50)),
    decode_head=dict(
        type='FCNHead', in_channels=1024, in_index=0, channels=1024),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False),
    ])
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)
