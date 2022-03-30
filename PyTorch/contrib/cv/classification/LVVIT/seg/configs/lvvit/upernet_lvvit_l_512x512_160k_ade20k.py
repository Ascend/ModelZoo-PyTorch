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

#coding=utf-8

_base_ = ['../_base_/datasets/ade20k.py','../_base_/default_runtime.py']
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='',
    backbone=dict(
        type='ViT',
        img_size=(448,448),
        depth=24,
        out_channels=2048,
        out_indices=(14, 17, 20, 23),
        patch_size=16,
        drop_path_rate=0.5,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=3.,
        qkv_bias=False,
        p_emb='4_2',
        stem_dim=128,
        use_side_layer=True),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
    # test_cfg=dict(mode='whole'))


optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, power=1.0, min_lr=0., by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
data=dict(samples_per_gpu=2)
fp16=dict()