
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
_base_ = './cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'
# model settings
model = dict(
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w40')),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))
