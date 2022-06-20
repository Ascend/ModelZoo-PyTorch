# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.config import CfgNode
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler

from .lr_scheduler import WarmupPolyLR


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        )
    else:
        return build_d2_lr_scheduler(cfg, optimizer)
