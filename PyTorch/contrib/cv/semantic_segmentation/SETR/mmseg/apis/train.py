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
import random

import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import IterBasedRunner, build_optimizer, Fp16OptimizerHook
from mmseg.core import DistEvalHook, EvalHook
# from mmseg.core import DistEvalTrainHook, EvalTrainHook
from mmseg.core import EvalTrainHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger

from apex import amp

def set_random_seed(seed, use_npu=False, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_npu == False:
        torch.cuda.manual_seed_all(seed)
    else :
        torch.npu.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=False) for ds in dataset
    ]
    # 添加amp的方法
    # cfg.use_amp = True 代表使用amp

    if cfg.use_amp == True:
        if distributed:
            model = model.cuda() if cfg.use_npu == False else model.npu()
        else:
            model = model.cuda(cfg.gpu_ids[0]) if cfg.use_npu == False else model.npu(cfg.gpu_ids[0])
        optimizer = build_optimizer(model, cfg.optimizer)
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.apex_opt_level,
                                    loss_scale=cfg.loss_scale_value)
        # amp._amp_state.loss_scalers[0]._loss_scale = 2**20

        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            print("*" * 3, "npu.current_device ", torch.npu.current_device())
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device() if cfg.use_npu == False else torch.npu.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.npu(), device_ids=cfg.gpu_ids)
    else: # 不使用apex amp
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            if cfg.use_npu == False:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDistributedDataParallel(
                    model.npu(),
                    device_ids=[torch.npu.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
        else:
            if cfg.use_npu == False:
                model = MMDataParallel(
                    model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
            else:
                model = MMDataParallel(
                    model.npu(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        optimizer = build_optimizer(model, cfg.optimizer)
        
        
    # build runner
    runner = IterBasedRunner(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)

    # 自定义hooks，根据优先级要求，就穿插进来。
    samples_per_gpu = cfg.data.samples_per_gpu
    workers_per_gpu = cfg.data.workers_per_gpu
    batch_size = len(cfg.gpu_ids) * samples_per_gpu
    num_workers = len(cfg.gpu_ids) * workers_per_gpu

    if not distributed or runner.rank==0:
        # runner.register_hook(EvalTrainHook(batch_size=batch_size, num_workers=num_workers))    
        train_eval_hook = EvalTrainHook(batch_size=batch_size, num_workers=num_workers)
    else: 
        train_eval_hook = None
    
    # fp16 setting
    if cfg.sys_fp_16:
        if cfg.loss_scale_value == None:
            fp16_cfg = dict(loss_scale='dynamic')
        else:
            fp16_cfg = dict(loss_scale=float(cfg.loss_scale_value))
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks( cfg.lr_config, optimizer_config,
                                    cfg.checkpoint_config, cfg.log_config,
                                    cfg.get('momentum_config', None),
                                    my_eval_train_hook=train_eval_hook)

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    runner.use_amp = cfg.use_amp
    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    
    if cfg.resume_from:
        runner.resume(cfg.resume_from, cfg.use_npu)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)