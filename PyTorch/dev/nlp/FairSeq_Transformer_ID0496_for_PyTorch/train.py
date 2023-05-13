#!/usr/bin/env python3 -u
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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random
import time
from apex import amp
import numpy as np
import torch

if torch.__version__ >= "1.8":
    import torch_npu
import os
import torch.distributed as dist
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

#hook
def hook_func(name, save_dict, module):
    def hook_function(module, inputs, outputs):
        # 逐层溢出检测
      a = torch.randn([8]).npu()
      float_status = torch.npu_alloc_float_status(a)
      torch.npu.synchronize()
      local_float_status = torch.npu_get_float_status(float_status)
      if (float_status.cpu()[0] != 0):
          torch.npu_clear_float_status(local_float_status)
          print(str(name) + " overflow!!!")

      # 逐层打印输入输出tensor的max，min，mean
      print(str(name))
      if inputs is not None and isinstance(inputs, torch.Tensor):
          in_temp = inputs.cpu().float()
          print("==== input ====", in_temp.max().item(), in_temp.mean().item(), in_temp.min().item())
      else:
          for input in inputs:
              if input is not None and isinstance(input, torch.Tensor):
                  in_temp = input.cpu().float()
                  print("===input====", in_temp.max().item(), in_temp.mean().item(), in_temp.min().item())
      if outputs is not None and isinstance(outputs, torch.Tensor):
          out_temp = outputs.cpu().float()
          print("===output===", out_temp.max().item(), out_temp.mean().item(), out_temp.min().item())
      else:
          for output in outputs:
              if output is not None and isinstance(output, torch.Tensor):
                  out_temp = output.cpu().float()
                  print("===output===", out_temp.max().item(), out_temp.mean().item(), out_temp.min().item())

    return hook_function

def main(args, init_distributed=False):
    # args.device_id = 5
    if not torch.cuda.is_available():
        a = torch.randn([8]).npu().fill_(2)
        float_status = torch.npu_alloc_float_status(a)
        local_float_status = torch.npu_get_float_status(float_status)
        torch.npu_clear_float_status(local_float_status)
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    '''
    npu_dict = {}
    for name, module in model.named_modules():
        module.register_forward_hook(hook_func('[forward]: '+name, npu_dict, module))
        module.register_backward_hook(hook_func('[backward]: '+name, npu_dict, module))
    '''
    criterion = task.build_criterion(args)
    #print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while (
        lr > args.min_lr
        and (epoch_itr.epoch < max_epoch or (epoch_itr.epoch == max_epoch
            and epoch_itr._next_epoch_itr is not None))
        and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        #if epoch_itr.epoch % args.save_interval == 0:
            #checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    '''
    input1 = torch.randn([1000, 1000, 300]).npu()
    input2 = torch.randn([1000, 1000, 300]).npu()
    input3 = torch.randn([1000, 1000, 300]).npu()
    input4 = torch.randn([1000, 1000, 300]).npu()
    input5 = torch.randn([1000, 1000, 300]).npu()
    input6 = torch.randn([1000, 1000, 300]).npu()
    input7 = torch.randn([1000, 1000, 300]).npu()
    input8 = torch.randn([1000, 1000, 300]).npu()
    input9 = torch.randn([1000, 1000, 300]).npu()
    input10 = torch.randn([1000, 1000, 300]).npu()
    input11 = torch.randn([1000, 1000, 300]).npu()
    input12 = torch.randn([1000, 1000, 300]).npu()
    input13 = torch.randn([1000, 1000, 300]).npu()
    input14 = torch.randn([1000, 1000, 300]).npu()
    input15 = torch.randn([1000, 1000, 300]).npu()
    input16 = torch.randn([1000, 1000, 300]).npu()
    input17 = torch.randn([1000, 1000, 300]).npu()
    input18 = torch.randn([1000, 1000, 300]).npu()
    '''
    if torch.__version__ >= "1.8":
        torch.npu.set_compile_mode(jit_compile=False)
    else:
        torch.npu.global_step_inc()
    num_steps = 0
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        #if i == 60:
        #    exit(0)
        #for tt in samples:
        #    print(tt["net_input"]["src_tokens"].size())
        start = time.time()
        if num_steps >= args.stop_step:
            if args.profiling == 'GE' or args.profiling == 'CANN':
                import sys
                sys.exit()
        elif num_steps < args.stop_step and num_steps >= args.start_step  and args.profiling == 'CANN':
            with torch.npu.profile(profiler_result_path="./CANN_prof"):
                if i<0 and epoch_itr.epoch == 1:
                    with torch.autograd.profiler.profile(use_npu = True) as prof:
                        log_output = trainer.train_step(samples)
                    #print(prof.key_averages().table())
                    prof.export_chrome_trace(("kdxf_transfomer_dynamic_step{}_npu.prof").format(i))
                else:
                    log_output = trainer.train_step(samples)
                end = time.time()
        elif num_steps <= args.stop_step and num_steps >= args.start_step and args.profiling == 'GE':
            with torch.npu.profile(profiler_result_path="./GE_prof"):
                if i<0 and epoch_itr.epoch == 1:
                    with torch.autograd.profiler.profile(use_npu = True) as prof:
                        log_output = trainer.train_step(samples)
                    #print(prof.key_averages().table())
                    prof.export_chrome_trace(("kdxf_transfomer_dynamic_step{}_npu.prof").format(i))
                else:
                    log_output = trainer.train_step(samples)
                end = time.time()
        else:
            if i<0 and epoch_itr.epoch == 1:
                with torch.autograd.profiler.profile(use_npu = True) as prof:
                    log_output = trainer.train_step(samples)
                #print(prof.key_averages().table())
                prof.export_chrome_trace(("kdxf_transfomer_dynamic_step{}_npu.prof").format(i))
            else:
                log_output = trainer.train_step(samples)
            '''
            if i >= 0 and i <=3:
                print("===== grad of step {} start".format(i))
                for name, parameters in trainer.model.named_parameters():
                    print(name, " {}, {}, {}".format(torch.max(parameters.grad).item(), torch.min(parameters.grad).item(),
                                                   torch.mean(parameters.grad).item()))
                print("===== grad of step {} over".format(i))
            else:
               exit(0)
            '''
            end = time.time()
            print("iteration {} time = {} ms".format(i, (end-start)*1000))
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = get_training_stats(trainer)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue  # these are already logged above
                if 'loss' in k or k == 'accuracy':
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)
                stats[k] = extra_meters[k].avg
            progress.log(stats, tag='train', step=stats['num_updates'])

            # ignore the first mini-batch in words-per-second and updates-per-second calculation
            if i == 0:
                trainer.get_meter('wps').reset()
                trainer.get_meter('ups').reset()

            num_updates = trainer.get_num_updates()
            if (
                not args.disable_validation
                and args.save_interval_updates > 0
                and num_updates % args.save_interval_updates == 0
                and num_updates > 0
            ):
                valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
                #checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            if num_updates >= max_update:
                break
        num_steps = num_steps + 1

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = args.distributed_rank = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    if args.bin :
        torch.npu.set_compile_mode(jit_compile=False)
    
    option = {}
    if args.precision_mode == 'must_keep_origin_dtype':
        torch.npu.config.allow_internal_format=False # 全局ND开关，默认值True
    if args.fp32:
        torch.npu.conv.allow_hf32 = False      # conv支持HF32开关，默认值True
        torch.npu.matmul.allow_hf32 = False   # matmul支持HF32开关，默认值True
    if not args.fp16:
        option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
    option["MM_BMM_ND_ENABLE"] = "enable"
    torch.npu.set_option(option)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        # assert args.distributed_world_size <= torch.cuda.device_count()
        # port = random.randint(10000, 20000)
        # args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        # args.distributed_rank = None  # set based on device id
        # if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
        #     print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        # torch.multiprocessing.spawn(
        #     fn=distributed_main,
        #     args=(args, ),
        #     nprocs=args.distributed_world_size,
        # )
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23456'
        assert args.distributed_world_size <= torch.npu.device_count()
        args.distributed_rank = args.device_id
        torch.npu.set_device(args.device_id)
        dist.init_process_group(backend=args.distributed_backend, world_size=args.distributed_world_size, rank=args.device_id)
        main(args)
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
