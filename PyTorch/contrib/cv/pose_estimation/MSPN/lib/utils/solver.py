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
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import torch.optim as optim
import apex

def make_optimizer(cfg, model, num_gpu):
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                lr=cfg.SOLVER.BASE_LR * num_gpu,
                betas=(0.9, 0.999), eps=1e-08,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    ########npu1p optimization 1 begin############
    elif cfg.SOLVER.OPTIMIZER == 'NpuFusedAdam':
        optimizer = apex.optimizers.NpuFusedAdam(model.parameters(),
                lr=cfg.SOLVER.BASE_LR * num_gpu,
                betas=(0.9, 0.999), eps=1e-08,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    ########npu1p optimization 1 end############
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    w_iters = cfg.SOLVER.WARMUP_ITERS
    w_fac = cfg.SOLVER.WARMUP_FACTOR
    max_iter = cfg.SOLVER.MAX_ITER
    lr_lambda = lambda iteration : w_fac + (1 - w_fac) * iteration / w_iters \
            if iteration < w_iters \
            else 1 - (iteration - w_iters) / (max_iter - w_iters)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    return scheduler

