# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Huawei Technologies Co., Ltd
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
# --------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import mmcv
import torch
import torch.nn as nn
from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           is_module_wrapper)

from mmseg import digit_version
from mmseg.utils import build_ddp, build_dp


def mock(*args, **kwargs):
    pass


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return self.conv(x)


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', mock)
def test_build_dp():
    model = Model()
    assert not is_module_wrapper(model)

    mmdp = build_dp(model, 'cpu')
    assert isinstance(mmdp, MMDataParallel)

    if torch.cuda.is_available():
        mmdp = build_dp(model, 'cuda')
        assert isinstance(mmdp, MMDataParallel)

    if digit_version(mmcv.__version__) >= digit_version('1.5.0'):
        from mmcv.device.mlu import MLUDataParallel
        from mmcv.utils import IS_MLU_AVAILABLE
        if IS_MLU_AVAILABLE:
            mludp = build_dp(model, 'mlu')
            assert isinstance(mludp, MLUDataParallel)


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', mock)
def test_build_ddp():
    model = Model()
    assert not is_module_wrapper(model)

    if torch.cuda.is_available():
        mmddp = build_ddp(
            model, 'cuda', device_ids=[0], process_group=MagicMock())
        assert isinstance(mmddp, MMDistributedDataParallel)

    if digit_version(mmcv.__version__) >= digit_version('1.5.0'):
        from mmcv.device.mlu import MLUDistributedDataParallel
        from mmcv.utils import IS_MLU_AVAILABLE
        if IS_MLU_AVAILABLE:
            mluddp = build_ddp(
                model, 'mlu', device_ids=[0], process_group=MagicMock())
            assert isinstance(mluddp, MLUDistributedDataParallel)
