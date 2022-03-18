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
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    params_dict = {}
    for param in params:
        p, l , w = param["params"], param["lr"], param["weight_decay"] 
        k = "{}_{}".format(l, w)
        
        if k not in params_dict:
            params_dict[k] = []
        params_dict[k].append(p[0])
    
    params = []
    for k in params_dict:
        lr, weight_decay = map(float, k.split("_"))
        params += [{"params": params_dict[k], "lr": lr, "weight_decay": weight_decay}]


    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        if  cfg.MODEL.DEVICE == "npu":
            from apex.optimizers import NpuFusedSGD
            print("optimizer_center use NpuFusedSGD")
            optimizer = NpuFusedSGD(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        if  cfg.MODEL.DEVICE == "npu":
            from apex.optimizers import NpuFusedAdam
            print("optimizer use NpuFusedAdam")
            optimizer = NpuFusedAdam(params)
        else:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)


    if  cfg.MODEL.DEVICE == "npu":
        from apex.optimizers import NpuFusedSGD
        print("optimizer_center use NpuFusedSGD")
        optimizer_center = NpuFusedSGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    else:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center
