#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import torch.optim as optim

def make_optimizer(cfg, model, num_gpu):
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                lr=cfg.SOLVER.BASE_LR * num_gpu,
                betas=(0.9, 0.999), eps=1e-08,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

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

