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
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        anchor_generator=dict(type='LegacyAnchorGenerator', center_offset=0.5),
        bbox_coder=dict(type='LegacyDeltaXYWHBBoxCoder'),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2,
                aligned=False)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2,
                aligned=False)),
        bbox_head=dict(
            bbox_coder=dict(type='LegacyDeltaXYWHBBoxCoder'),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    rpn_proposal=dict(nms_post=2000, max_num=2000),
    rcnn=dict(assigner=dict(match_low_quality=True)))
