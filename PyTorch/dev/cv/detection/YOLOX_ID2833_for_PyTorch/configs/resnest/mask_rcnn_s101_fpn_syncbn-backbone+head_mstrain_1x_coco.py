
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
_base_ = './mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py'
model = dict(
    backbone=dict(
        stem_channels=128,
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://resnest101')))
