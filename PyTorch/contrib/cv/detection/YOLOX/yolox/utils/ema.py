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


import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.is_fused = False

    def update(self, model, device='npu:0', model_params_fused=None):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            if 'npu' in str(device) and True:
                from apex.contrib.combine_tensors import combine_npu
                d_inv = 1. - d
                d = torch.tensor([d], device=device)
                msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
                if not self.is_fused:
                    pg0, pg1, pg2 = [], [], [] # optimizer parameters groups

                    # this process needs special attention, the order of params should be identical to model
                    for name, p in self.ema.named_parameters():
                        if p.dtype.is_floating_point:
                            if '.bias' in name:
                                pg2.append(p) # biases
                            elif '.weight' in name and '.bn' not in name:
                                pg1.append(p) # apply weight decay
                            else:
                                pg0.append(p) # all else
                    ema_all_params = pg0 + pg1 + pg2
                    self.ema_params_fused = combine_npu(ema_all_params)
                    # print('----- self.ema_params_fused: ', self.ema_params_fused.shape)

                    ema_all_buffers = []
                    for name, b in self.ema.named_buffers():
                        if b.dtype.is_floating_point:
                            ema_all_buffers.append(b)
                        else:
                            continue
                    self.ema_buffers_fused = combine_npu(ema_all_buffers)

                    model_all_buffers = []
                    for name, b in model.named_buffers():
                        if b.dtype.is_floating_point:
                            model_all_buffers.append(b)
                        else:
                            continue
                    self.model_buffers_fused = combine_npu(model_all_buffers)

                    self.is_fused = True

                self.ema_params_fused *= d
                self.ema_params_fused.add_(model_params_fused, alpha=d_inv)

                self.ema_buffers_fused *= d
                self.ema_buffers_fused.add_(self.model_buffers_fused, alpha=d_inv)

            else:
                print('EMA updated without combined tensor.')
                msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
                for k, v in self.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        v *= d
                        v += (1. - d) * msd[k].detach()

