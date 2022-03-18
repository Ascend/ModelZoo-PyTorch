# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
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
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# high_resoluton_net related params for classification
POSE_HIGH_RESOLUTION_NET = CN()
POSE_HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGH_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
POSE_HIGH_RESOLUTION_NET.WITH_HEAD = True

POSE_HIGH_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
POSE_HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
POSE_HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
POSE_HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'cls_hrnet': POSE_HIGH_RESOLUTION_NET,
}
