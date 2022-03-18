# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from fvcore.common.config import CfgNode as _CfgNode
from module.utils import PathManager

class CN(_CfgNode):
    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

CfgNode = CN

_C = CN()

_C.SEED = 1

## data related
_C.DATA = CN()
_C.DATA.DATASET = 'ImageNet'
# assuming you've set up the dataset using provided script
_C.DATA.ROOT = '/opt/npu/imagenet'
_C.DATA.BASE_SIZE = None
_C.DATA.CROP_SIZE = 224
_C.DATA.LABEL_SMOOTHING = 0.0
_C.DATA.MIXUP = 0.0
_C.DATA.RAND_AUG = False

## model related
_C.MODEL = CN()
_C.MODEL.NAME = 'resnest50'
_C.MODEL.FINAL_DROP = False
_C.MODEL.NUM_CLASSES = 1000

## training params
_C.TRAINING = CN()
# (per-gpu batch size)
_C.TRAINING.BATCH_SIZE = 64
_C.TRAINING.TEST_BATCH_SIZE = 256
_C.TRAINING.LAST_GAMMA = False
_C.TRAINING.EPOCHS = 120
_C.TRAINING.START_EPOCHS = 0
_C.TRAINING.WORKERS = 4

## optimizer params
_C.OPTIMIZER = CN()
# (per-gpu lr)
_C.OPTIMIZER.LR = 0.025
_C.OPTIMIZER.LR_SCHEDULER = 'cos'
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
_C.OPTIMIZER.DISABLE_BN_WD = False
_C.OPTIMIZER.WARMUP_EPOCHS = 0

def get_cfg() -> CN:
    return _C.clone()
