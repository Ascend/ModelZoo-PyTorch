# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
import unittest

import torch
from torch import nn

sys.path.append('.')
from solver.lr_scheduler import WarmupMultiStepLR
from solver.build import make_optimizer
from config import cfg


class MyTestCase(unittest.TestCase):
    def test_something(self):
        net = nn.Linear(10, 10)
        optimizer = make_optimizer(cfg, net)
        lr_scheduler = WarmupMultiStepLR(optimizer, [20, 40], warmup_iters=10)
        for i in range(50):
            lr_scheduler.step()
            for j in range(3):
                print(i, lr_scheduler.get_lr()[0])
                optimizer.step()


if __name__ == '__main__':
    unittest.main()
