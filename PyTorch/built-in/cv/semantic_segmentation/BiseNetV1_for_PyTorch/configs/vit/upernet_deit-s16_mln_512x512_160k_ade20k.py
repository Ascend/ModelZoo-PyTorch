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
_base_ = './upernet_vit-b16_mln_512x512_160k_ade20k.py'

model = dict(
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[384, 384, 384, 384]),
    neck=dict(in_channels=[384, 384, 384, 384], out_channels=384),
    auxiliary_head=dict(num_classes=150, in_channels=384))
