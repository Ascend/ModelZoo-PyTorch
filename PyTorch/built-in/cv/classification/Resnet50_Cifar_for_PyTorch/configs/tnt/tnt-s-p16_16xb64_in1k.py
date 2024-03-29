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
# accuracy_top-1 : 81.52 accuracy_top-5 : 95.73
_base_ = [
    '../_base_/models/tnt_s_patch16_224.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(248, -1),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=64, workers_per_gpu=4, test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=300)
