#     Copyright 2021 Huawei
#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# model settings
model = dict(
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(norm_cfg=dict(type='BN', requires_grad=True)))
checkpoint_config = dict(interval=2000)
data = dict(samples_per_gpu=6, workers_per_gpu=32)
runner = dict(max_iters=7000)
evaluation = dict(interval=1000)

optimizer = dict(type='NpuFusedSGD', lr=0.06)
optimizer_config = dict(type='AmpOptimizerHook')
opt_level = 'O1'
loss_scale = 256

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook_FPS', by_epoch=False),
    ])

