# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_100k_iters.py',
    '../../_base_/det_models/dbnetpp_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/synthtext.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline_r50dcnv2 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg_r50dcnv2),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]],
        clip_invalid_ploys=False),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]

test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))

evaluation = dict(interval=200000, metric='hmean-iou')  # do not evaluate
