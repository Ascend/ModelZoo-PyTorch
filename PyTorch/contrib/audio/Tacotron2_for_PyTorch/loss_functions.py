# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from tacotron2.loss_function import Tacotron2Loss
from waveglow.loss_function import WaveGlowLoss


def get_loss_function(loss_function, sigma=1.0):
    if loss_function == 'Tacotron2':
        loss = Tacotron2Loss()
    elif loss_function == 'WaveGlow':
        loss = WaveGlowLoss(sigma=sigma)
    else:
        raise NotImplementedError(
            "unknown loss function requested: {}".format(loss_function))

    loss.npu()
    return loss
