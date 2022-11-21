
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
_base_ = './cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py'  # noqa: E501
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
