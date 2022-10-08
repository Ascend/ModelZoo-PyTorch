# Copyright 2022 Huawei Technologies Co., Ltd.
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
_base_ = 'solov2_light_r50_fpn_3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        depth=34, init_cfg=dict(checkpoint='torchvision://resnet34')),
    neck=dict(in_channels=[64, 128, 256, 512]))
