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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                       x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                       torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res
