# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
import numpy as np
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import densetorch as dt

from network import get_encoder_and_decoder_params


def get_lr_schedulers(
    enc_optim,
    dec_optim,
    enc_lr_gamma,
    dec_lr_gamma,
    enc_scheduler_type,
    dec_scheduler_type,
    epochs_per_stage,
):
    milestones = np.cumsum(epochs_per_stage)
    max_epochs = milestones[-1]
    schedulers = [
        dt.misc.create_scheduler(
            scheduler_type=enc_scheduler_type,
            optim=enc_optim,
            gamma=enc_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
        dt.misc.create_scheduler(
            scheduler_type=dec_scheduler_type,
            optim=dec_optim,
            gamma=dec_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
    ]
    return schedulers


def get_optimisers(
    model,
    enc_optim_type,
    enc_lr,
    enc_weight_decay,
    enc_momentum,
    dec_optim_type,
    dec_lr,
    dec_weight_decay,
    dec_momentum,
):
    enc_params, dec_params = get_encoder_and_decoder_params(model)
    optimisers = [
        create_SGD(
            parameters=enc_params,
            lr=enc_lr,
            weight_decay=enc_weight_decay,
            momentum=enc_momentum,
        ),
        create_SGD(
            parameters=dec_params,
            lr=dec_lr,
            weight_decay=dec_weight_decay,
            momentum=dec_momentum,
        ),
    ]
    return optimisers

def create_SGD(parameters, lr, weight_decay, momentum):
    # only for SGD
    import apex
    return apex.optimizers.NpuFusedSGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum) 
    