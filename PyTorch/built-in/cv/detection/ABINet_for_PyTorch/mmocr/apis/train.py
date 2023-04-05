# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist

try:
    import apex
    from apex import amp
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset

from mmocr import digit_version
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.utils import get_root_logger


class ApexOptimizerHook(OptimizerHook):

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
            scaled_loss.backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    default_loader_cfg = {
        **dict(
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.get('seed'),
            drop_last=False,
            pin_memory=True,
            persistent_workers=False),
        **({} if torch.__version__ != 'parrots' else dict(
            prefetch_num=2,
        )),
    }
    # update overall dataloader(for train, val and test) setting
    default_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(default_loader_cfg,
                            **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if torch.npu.is_available():
            model = NPUDistributedDataParallel(
                model.npu(),
                device_ids=[torch.npu.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        if torch.npu.is_available():
            model = NPUDataParallel(model.npu(), device_ids=cfg.gpu_ids)
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    if torch.npu.is_available():
        optimizer = apex.optimizers.NpuFusedAdam(model.npu().module.parameters(),
                                                 lr=cfg.optimizer['lr'])
        model.module, optimizer = amp.initialize(model.module, optimizer,
                                                 opt_level='O1',loss_scale=1024.,
                                                 combine_grad=True)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    else:
        optimizer_config = ApexOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_samples_per_gpu = (cfg.data.get('val_dataloader', {})).get(
            'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
        if val_samples_per_gpu > 1:
            # Support batch_size > 1 in test for text recognition
            # by disable MultiRotateAugOCR since it is useless for most case
            cfg = disable_text_recog_aug_test(cfg)
            cfg = replace_image_to_tensor(cfg)

        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_loader_cfg = {
            **default_loader_cfg,
            **dict(shuffle=False, drop_last=False),
            **cfg.data.get('val_dataloader', {}),
            **dict(samples_per_gpu=val_samples_per_gpu)
        }

        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed. If the seed is None, it will be replaced by a
    random number, and then broadcasted to all processes.

    Args:
        seed (int, Optional): The seed.
        device (str): The device where the seed will be put on.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
