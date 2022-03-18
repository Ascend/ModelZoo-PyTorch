# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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


import unittest
import torch
from torch import nn

from detectron2.engine import SimpleTrainer


class SimpleModel(nn.Sequential):
    def forward(self, x):
        return {"loss": x.sum() + sum([x.mean() for x in self.parameters()])}


class TestTrainer(unittest.TestCase):
    def test_simple_trainer(self, device="cpu"):
        device = torch.device(device)
        model = SimpleModel(nn.Linear(10, 10)).to(device)

        def data_loader():
            while True:
                yield torch.rand(3, 3).to(device)

        trainer = SimpleTrainer(model, data_loader(), torch.optim.SGD(model.parameters(), 0.1))
        trainer.train(0, 10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_trainer_cuda(self):
        self.test_simple_trainer(device="cuda")
