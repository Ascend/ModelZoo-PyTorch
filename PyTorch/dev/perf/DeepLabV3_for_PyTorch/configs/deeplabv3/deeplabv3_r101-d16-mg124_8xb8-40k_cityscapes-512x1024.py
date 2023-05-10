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

_base_ = './deeplabv3_r101-d16-mg124_4xb2-40k_cityscapes-512x1024.py'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
)
test_dataloader = val_dataloader

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg),
    decode_head=dict(
        dropout_ratio=0.0,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        dropout_ratio=0.0,
        norm_cfg=norm_cfg,
    )
)

# optimizer
optimizer = dict(type='NpuFusedSGD', lr=0.08, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
