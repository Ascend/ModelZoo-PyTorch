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

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='GHMC',
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='GHMR',
            mu=0.02,
            bins=10,
            momentum=0.7,
            loss_weight=10.0)))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
