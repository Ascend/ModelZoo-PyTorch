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
_base_ = '../retinanet/retinanet_x101_32x4d_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        type='PISARetinaHead',
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))

train_cfg = dict(isr=dict(k=2., bias=0.), carl=dict(k=1., bias=0.2))
