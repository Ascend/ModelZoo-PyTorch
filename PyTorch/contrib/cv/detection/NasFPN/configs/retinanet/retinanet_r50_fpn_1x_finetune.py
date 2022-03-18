# Copyright 2020 Huawei Technologies Co., Ltd
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

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='NpuFusedSGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
log_config = dict( # config to register logger hook
    interval=50, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook')
    ])

dist_params = dict(backend='hccl')

model = dict(
    bbox_head=dict(
    num_classes=5
    ))

classes = ('person', 'car', 'cat', 'dog', 'train')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes)
)

load_from = './work_dirs/retinanet_r50_fpn_1x_coco/epoch_12.pth'