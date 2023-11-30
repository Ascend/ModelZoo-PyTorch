# Copyright 2021 Huawei Technologies Co., Ltd
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

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='NpuFusedSGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
log_config = dict( # config to register logger hook
    interval=10, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook')
    ])

dist_params = dict(backend='hccl')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=20
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))