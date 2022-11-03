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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torchvision.transforms as transforms

import numpy as np
import random
import cv2

# --------------------------------------------------------------------------   
class LambdaMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=32, layer_dim=16):
        super(LambdaMLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1_mu = nn.Linear(self.input_dim, layer_dim)
        self.fc2_mu = nn.Linear(layer_dim, layer_dim)
        self.fc3_mu = nn.Linear(layer_dim, self.output_dim)
        self.bn1_mu = nn.BatchNorm1d(layer_dim)
        self.bn2_mu = nn.BatchNorm1d(layer_dim)

        self.fc1_sigma = nn.Linear(self.input_dim, layer_dim)
        self.fc2_sigma = nn.Linear(layer_dim, layer_dim)
        self.fc3_sigma = nn.Linear(layer_dim, self.output_dim)
        self.bn1_sigma = nn.BatchNorm1d(layer_dim)
        self.bn2_sigma = nn.BatchNorm1d(layer_dim)

        return

    def forward(self, lambda_vec):
        x = F.relu(self.bn1_mu(self.fc1_mu(lambda_vec)))
        x = F.relu(self.bn2_mu(self.fc2_mu(x)))
        mu = self.fc3_mu(x)

        x = F.relu(self.bn1_sigma(self.fc1_sigma(lambda_vec)))
        x = F.relu(self.bn2_sigma(self.fc2_sigma(x)))
        sigma = self.fc3_sigma(x)

        return mu, sigma

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # -------------------------
    G = SmplGenerator().cuda()
    G = RotationGenerator().cuda()

    batch_size = 4
    z = torch.Tensor(np.random.normal(0, 1, (batch_size, 10))).cuda()
    import pdb; pdb.set_trace()