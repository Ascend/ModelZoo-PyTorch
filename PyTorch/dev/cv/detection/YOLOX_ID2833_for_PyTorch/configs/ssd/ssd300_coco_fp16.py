
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
_base_ = ['./ssd300_coco.py']

fp16 = dict(loss_scale=512.)

# learning policy
# In order to avoid non-convergence in the early stage of
# mixed-precision training, the warmup in the lr_config is set to linear,
# warmup_iters increases and warmup_ratio decreases.
lr_config = dict(warmup='linear', warmup_iters=1000, warmup_ratio=1.0 / 10)
