# Copyright 2023 Huawei Technologies Co., Ltd
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

# import torch
# from mpu.transformer import GPT2ParallelSelfAttention
#
# b = torch.arange(2) * 1000
# h = torch.arange(3) * 100
# pos_seq = torch.arange(9, -1, -1)
# query = torch.arange(7) * 10
# s = pos_seq.unsqueeze(0) + query.unsqueeze(1)
# s = b.view(-1, 1, 1, 1) + h.view(1, -1, 1, 1) + s
# s = GPT2ParallelSelfAttention._rel_shift(s)
# print(s)

from torch.nn.modules import Linear
from torch.optim import Adam
from learning_rates import AnnealingLR
import matplotlib.pyplot as plt
import numpy as np


def main():
    model = Linear(10, 10)
    optimizer = Adam(model.parameters())
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=0.00015,
                               warmup_iter=3000,
                               num_iters=300000,
                               decay_style='cosine',
                               decay_ratio=0.1)
    steps = np.arange(0, 400000, 10, dtype=np.long)
    rates = []
    for step in steps:
        lr_scheduler.num_iters = step
        rates.append(lr_scheduler.get_lr())
    print(rates)
    plt.plot(steps, rates)
    plt.savefig("lr.pdf", format='pdf')