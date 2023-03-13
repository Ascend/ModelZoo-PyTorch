# Copyright 2022 Huawei Technologies Co., Ltd
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
from copy import deepcopy
import os

import torch

class ModelEmaV2Npu(torch.nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2Npu, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.is_fused = False

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, model_params_fused):
        with torch.no_grad():
            if not self.is_fused:
                if str(os.environ['use_amp']) == 'apex':
                    from apex.contrib.combine_tensors import combine_npu
                elif str(os.environ['use_amp']) == 'native':
                    from torch_npu.utils import npu_combine_tensors as combine_npu
                decay = []
                no_decay = []
                for name, param in self.module.named_parameters():
                    if not param.requires_grad:
                        continue
                    if len(param.shape) == 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)
                ema_all_params = no_decay + decay
                self.ema_params_fused = combine_npu(ema_all_params)

                ema_all_buffers = []
                for name, b in self.module.named_buffers():
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
            
            self.ema_params_fused *= self.decay
            self.ema_params_fused.add_(model_params_fused, alpha=1 - self.decay)

            self.ema_buffers_fused *= self.decay
            self.ema_buffers_fused.add_(self.model_buffers_fused, alpha=1 - self.decay)           

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)