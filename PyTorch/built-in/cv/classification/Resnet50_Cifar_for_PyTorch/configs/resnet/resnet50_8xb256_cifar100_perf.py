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
_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/default_runtime.py'
]

batch_size = 256

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=1.0, num_classes=100, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=100, prob=0.5)
    ]))

optimizer = dict(type='SGD', lr=0.1 * (batch_size / 16), momentum=0.9, weight_decay=0.0005)

#optimizer
optimizer_config = dict(grad_clip=None)
#learning policy
runner = dict(type='EpochBasedRunner', max_epochs=2)
lr_config = dict(policy='CosineAnnealing', min_lr=0,
                 warmup='linear',
                 warmup_ratio=1e-3,
                 warmup_iters=20,
                 warmup_by_epoch=True)

data = dict(
    samples_per_gpu=batch_size, 
    workers_per_gpu=12,
    persistent_workers=True,
)

log_config = dict(interval=1)