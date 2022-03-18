# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import

import torch
import apex


def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return apex.optimizers.NpuFusedAdam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return apex.optimizers.NpuFusedSGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))
