#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# -------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import collections
import itertools
import os
import math
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import time
import ctypes

import sys
import threading

from copy import deepcopy
from utils import distributed_utils, options, utils
from utils.ddp_trainer import DDPTrainer
from utils.meters import StopwatchMeter, TimeMeter
import data
from data import tokenizer, dictionary, data_utils, load_dataset_splits
from models import build_model
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import dllogger as DLLogger
from utils.log_helper import AggregatorBackend, setup_logger


NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
RANK = int(os.getenv('RANK'))
torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)
MAX = 2147483647

def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)
seed_shape = (32 * 1024 * 12, )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    print(args)
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    #mp.spawn(main_worker, nprocs=args.distributed_world_size, args=(args.distributed_world_size, args))
    main_worker(pid_idx=RANK, args=args)



def main_worker(pid_idx, args):
    setup_logger(args)
    print('pid_idx:',str(pid_idx))
    args.distributed_rank = pid_idx
    args.device_id = args.distributed_rank
    #dist.init_process_group(backend=args.dist_backend, world_size=NPU_WORLD_SIZE, rank=args.distributed_rank)
    loc = 'npu:{}'.format(args.device_id)
    torch.npu.set_device(loc)

    if args.max_tokens is None:
        args.max_tokens = 6000

    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    add_extra_items_to_checkpoint({'src_dict': src_dict, 'tgt_dict': tgt_dict})
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(loc)
    model = build_model(args, seed=seed)
    if args.distributed_world_size > 1 :
        print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    trainer = DDPTrainer(args, model)
    if args.distributed_world_size > 1 :
        print('| model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))
        print('| training on {} NPUs'.format(args.distributed_world_size))

    if args.distributed_world_size > 1 :
        print('| max sentences per NPU = {}'.format(args.max_sentences))

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        max_positions_num=96,

    )
    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    run_summary = {'loss': float('inf'),
                   'val_loss': float('inf'),
                   'speed': 0,
                   'accuracy': 0}

    # max_update
    m=0
    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        m=m+1
        if m >=2:pass
        DLLogger.log(step=trainer.get_num_updates(), data={'epoch': epoch_itr.epoch}, verbosity=0)
        # train for one epoch
        train(args, trainer, datasets, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, datasets, valid_subsets)
            DLLogger.log(step=trainer.get_num_updates(), data={'val_loss': valid_losses[0]},
                         verbosity=1)


        if valid_losses[0] < run_summary['val_loss']:
            run_summary['val_loss'] = valid_losses[0]
        run_summary['loss'] = valid_losses[0]
        run_summary['speed'] = trainer.throughput_meter.u_avg

        # Only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # Save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    train_meter.stop()
    DLLogger.log(step=[], data=run_summary, verbosity=0)
    DLLogger.log(step='RUN', data={'walltime': train_meter.sum}, verbosity=0)
    if args.distributed_world_size > 1 :
        print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, datasets, epoch_itr):
    """Train the model for one epoch."""

    itr = epoch_itr.next_epoch_itr()

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    num_batches = len(epoch_itr)

    batch_time = AverageMeter('Time', ':6.3f')
    sentence_s = AverageMeter('Sentence/s', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(int(num_batches/args.distributed_world_size/update_freq),
                             [batch_time, sentence_s,losses],
                             prefix = "Epoch: [{}]".format(epoch_itr.epoch))

    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf


    # reset meters
    DLLogger.flush()
    trainer.get_throughput_meter().reset()
   
    for i, sample in enumerate(itr):
        if i>100:pass
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            loss = trainer.train_step(sample, update_params=False, last_step=(i == len(itr) - 1))
            continue
        else:
            loss = trainer.train_step(sample, update_params=True, last_step=(i == len(itr) - 1))
            if loss != None:
                losses.update(loss)
        # Execute torch.npu.empty_cache() to avoid oom on PT1.11
        if i == 3:
            torch.npu.empty_cache()
        if i >= 4:
            t = time.time()
            batch_time.update((t - end)/update_freq)
            sentence_s.update(args.max_sentences/(t-end)*args.distributed_world_size)
            end = time.time()
        if i < 4:
            end = time.time()
        if i >= 4:
            if args.distributed_world_size > 1 :
                progress.display(int((i+1)/update_freq))


        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_throughput_meter().reset()
            for backend in DLLogger.GLOBAL_LOGGER.backends:
                if isinstance(backend, AggregatorBackend):
                    backend._reset_perf_meter('tokens')
                    backend._reset_perf_meter('updates')
                    break

        # Mid epoch checkpoint
        num_updates = trainer.get_num_updates()
        if args.distributed_world_size > 1 :
            if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
                valid_losses = validate(args, trainer, datasets, [first_valid])
                save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            if (i + 1) % args.log_interval == 0:
                DLLogger.flush()

        if num_updates >= max_update:
            break

    if args.distributed_world_size > 1 :
        if batch_time.avg > 0:
            print("End of epoch, batch_size:", args.max_sentences, 'Time: {:.3f}'.format(batch_time.avg),
                  ' Sentence/s@all {:.3f}'.format(
                      args.max_sentences / batch_time.avg * args.distributed_world_size))

    # Print epoch stats and reset training meters
    if args.distributed_world_size > 1 :
        DLLogger.log(step=trainer.get_num_updates(), data={'speed': trainer.get_throughput_meter().avg}, verbosity=0)
        DLLogger.flush()


def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    # Reset value iterations counter
    trainer._num_val_iterations = 0

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=datasets[subset],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=args.max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            max_positions_num=1024,
        ).next_epoch_itr(shuffle=False)

        # reset validation loss meters
        if args.distributed_world_size > 1 :
            DLLogger.flush()

        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        subset_loss = sum(subset_losses) / len(subset_losses)

        DLLogger.flush()

        valid_losses.append(subset_loss)
        if args.distributed_world_size > 1 :
            print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses



def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(os.path.join(args.save_dir, 'checkpoints'),
                                             pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def add_extra_items_to_checkpoint(dict):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(dict)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])
            if args.distributed_world_size > 1 :
                print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                    checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


if __name__ == '__main__':
    main()
