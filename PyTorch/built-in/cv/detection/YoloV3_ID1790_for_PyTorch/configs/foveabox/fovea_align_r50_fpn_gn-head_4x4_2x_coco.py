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

_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
model = dict(
    bbox_head=dict(
        with_deform=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
