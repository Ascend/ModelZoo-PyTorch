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
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger

# weik start add
from apex import amp
from apex.optimizers import NpuFusedSGD
# weik end add

def set_random_seed(seed, deterministic=False):
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
    torch.cuda.manual_seed_all(seed)
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
            pin_memory=False,
            drop_last=True) for ds in dataset
    ]

    # weik start add
    optimizer = build_optimizer(model, cfg.optimizer)

    # put model on gpus
    if cfg.device == 'npu':
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = model.npu()
            if cfg.amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.opt_level, loss_scale=cfg.loss_scale, combine_grad=True)
            model = MMDistributedDataParallel(
                model.npu(),
                device_ids=[torch.npu.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = model.npu()
            if cfg.amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.opt_level, loss_scale=cfg.loss_scale, combine_grad=True)
            model = MMDataParallel(
                model.npu(), device_ids=cfg.gpu_ids)
    elif cfg.device == 'gpu':
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = model.cuda()
            if cfg.amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=None)
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = model.cuda(cfg.gpu_ids[0])
            if cfg.amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = MMDataParallel(
                model, device_ids=cfg.gpu_ids)
    else:
        print("error: devices should be gpu or npu.")

    if cfg.prof:
        profiling(data_loaders[0], model, optimizer, cfg, buffstep=20)
        return
    # weik add end

    # weik comment start
    # # put model on gpus
    # if distributed:
    #     find_unused_parameters = cfg.get('find_unused_parameters', False)
    #     # Sets the `find_unused_parameters` parameter in
    #     # torch.nn.parallel.DistributedDataParallel
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #         find_unused_parameters=find_unused_parameters)
    # else:
    #     model = MMDataParallel(
    #         model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    #
    # # build runner
    # optimizer = build_optimizer(model, cfg.optimizer)
    # weik comment end

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            # NPU - weik
            samples_per_gpu=cfg.data.samples_per_gpu,
            num_of_gpus=len(cfg.gpu_ids)
            ))

    # weik add start
    if distributed:
        runner.number_device = int(torch.npu.device_count())
    else:
        runner.number_device = 1
    runner.batch_size = data_loaders[0].batch_size

    if cfg.amp:
        runner.amp = True
    # weik add end

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

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
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if cfg.prof_test :
        profiling_test(val_dataloader, model, cfg, buffstep=5)
        return


    runner.run(data_loaders, cfg.workflow)


def profiling(data_loader, model, optimizer, args, buffstep=5):
    # switch to train mode
    model.train()
    total_batch = len(data_loader)

    def update(step, model, data_batch, optimizer):
        outputs = model.train_step(data_batch, optimizer)
        print("Iter [{}/{}] lr: {}, loss: {}".format(step+1, total_batch, optimizer.param_groups[0]['lr'], outputs['loss']))
        if args.amp:
            with(amp.scale_loss(outputs['loss'], optimizer)) as scaled_loss:
                scaled_loss.backward()
        else:
            outputs['loss'].backward()
        optimizer.step()


    for step, data_batch in enumerate(data_loader):
        # if args.device == 'npu':
        #     loc = 'npu:{}'.format(args.gpu_ids)
        #     data_batch = data_batch.to(loc, non_blocking=True).to(torch.float)
        #     target = target.to(torch.int32).to(loc, non_blocking=True)
        # else:
        #     data_batch = data_batch.cuda(args.gpu_ids, non_blocking=True)

        if step < buffstep:
            update(step, model, data_batch, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(step, model, data_batch, optimizer)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(step, model, data_batch, optimizer)
            break

    prof.export_chrome_trace("output.prof")
    print('prof: done\n')


def profiling_test(data_loader, model, args, buffstep=5):
    # switch to train mode
    model.eval()
    total_batch = len(data_loader)

    def update(step, model, data_batch):
        with torch.no_grad():
            result = model(return_loss=False, **data_batch)
        print("Iter [{}/{}]".format(step+1, total_batch))


    for step, data_batch in enumerate(data_loader):
        # if args.device == 'npu':
        #     loc = 'npu:{}'.format(args.gpu_ids)
        #     data_batch = data_batch.to(loc, non_blocking=True).to(torch.float)
        #     target = target.to(torch.int32).to(loc, non_blocking=True)
        # else:
        #     data_batch = data_batch.cuda(args.gpu_ids, non_blocking=True)
        if step < buffstep:
            update(step, model, data_batch)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(step, model, data_batch)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(step, model, data_batch)
            break

    prof.export_chrome_trace("output.prof")
    print('prof: done\n')
