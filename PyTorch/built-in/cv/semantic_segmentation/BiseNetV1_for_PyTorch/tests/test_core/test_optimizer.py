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
import pytest
import torch
import torch.nn as nn
from mmcv.runner import DefaultOptimizerConstructor

from mmseg.core.builder import (OPTIMIZER_BUILDERS, build_optimizer,
                                build_optimizer_constructor)


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def test_build_optimizer_constructor():
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optim_constructor_cfg = dict(
        type='DefaultOptimizerConstructor', optimizer_cfg=optimizer_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    # Test whether optimizer constructor can be built from parent.
    assert type(optim_constructor) is DefaultOptimizerConstructor

    @OPTIMIZER_BUILDERS.register_module()
    class MyOptimizerConstructor(DefaultOptimizerConstructor):
        pass

    optim_constructor_cfg = dict(
        type='MyOptimizerConstructor', optimizer_cfg=optimizer_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    # Test optimizer constructor can be built from child registry.
    assert type(optim_constructor) is MyOptimizerConstructor

    # Test unregistered constructor cannot be built
    with pytest.raises(KeyError):
        build_optimizer_constructor(dict(type='A'))


def test_build_optimizer():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optimizer = build_optimizer(model, optimizer_cfg)
    # test whether optimizer is successfully built from parent.
    assert isinstance(optimizer, torch.optim.SGD)
