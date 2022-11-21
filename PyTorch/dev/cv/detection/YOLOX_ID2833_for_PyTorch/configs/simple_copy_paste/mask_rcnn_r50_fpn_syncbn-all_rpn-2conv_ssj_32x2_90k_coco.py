
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
_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco.py'

# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
lr_config = dict(
    warmup_iters=500, warmup_ratio=0.067, step=[81000, 85500, 87750])
# 90k iterations with batch_size 64 is roughly equivalent to 48 epochs
runner = dict(type='IterBasedRunner', max_iters=90000)
