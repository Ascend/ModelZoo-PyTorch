# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
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
import argparse

import torch

from models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
from utils.criterions import CRITERION_REGISTRY
from optim import OPTIMIZER_REGISTRY
from optim.lr_scheduler import LR_SCHEDULER_REGISTRY


def get_training_parser():
    parser = get_parser('Trainer')
    add_dataset_args(parser, train=True, gen=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_inference_parser():
    parser = get_parser('Generation')
    add_dataset_args(parser, gen=True)
    return parser


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def parse_args_and_arch(parser, input_args=None, parse_known=False):
    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)


    # Add model-specific args to parser.
    if hasattr(args, 'arch'):
        model_specific_group = parser.add_argument_group(
            'Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add *-specific args to parser.
    if hasattr(args, 'optimizer'):
        OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    if hasattr(args, 'lr_scheduler'):
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Post-process args.
    if hasattr(args, 'lr'):
        args.lr = eval_str_list(args.lr, type=float)
    if hasattr(args, 'update_freq'):
        args.update_freq = eval_str_list(args.update_freq, type=int)
    if hasattr(args, 'max_sentences_valid') and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences

    args.max_positions = (args.max_source_positions, args.max_target_positions)

    # Apply architecture configuration.
    if hasattr(args, 'arch'):
        ARCH_CONFIG_REGISTRY[args.arch](args)

    # Override args for profiling
    if hasattr(args, 'profile') and args.profile:
        args.max_update = args.profiler_steps

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc):
    parser = argparse.ArgumentParser(
        description='Facebook AI Research Sequence-to-Sequence Toolkit -- ' + desc)
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='print aggregated stats and flush json log every N iteration')
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--amp-level', type=str, default="O1", help='choose apm\'s optimization level')
    parser.add_argument('--stat-file', type=str, default='run_log.json', help='Name of the file containing DLLogger output')
    parser.add_argument('--save-dir', metavar='DIR', default='results',
                       help='path to save checkpoints and logs')
    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group('Dataset and data loading')
    group.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                       help='ignore too long or too short lines in valid and test set')
    group.add_argument('--max-tokens', type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    group.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                        help='source language')
    parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                        help='target language')
    parser.add_argument('--raw-text', action='store_true',
                        help='load raw text dataset')
    parser.add_argument('--left-pad-source', default=True, type=bool, metavar='BOOL',
                        help='pad the source on the left (default: True)')
    parser.add_argument('--left-pad-target', default=False, type=bool, metavar='BOOL',
                        help='pad the target on the left (default: False)')
    parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                        help='max number of tokens in the source sequence')
    parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                        help='max number of tokens in the target sequence')
    parser.add_argument('--pad-sequence', default=1, type=int, metavar='N',
                            help='Pad sequences to a multiple of N')
    if train:
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        group.add_argument('--train-subset', default='train', metavar='SPLIT',
                           choices=['train', 'valid', 'test'],
                           help='data subset to use for training (train, valid, test)')
        group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                           help='comma separated list of data subsets to use for validation'
                                ' (train, valid, valid1, test, test1)')
        group.add_argument('--max-sentences-valid', type=int, metavar='N',
                           help='maximum number of sentences in a validation batch'
                                ' (defaults to --max-sentences)')
    if gen:
        group.add_argument('--gen-subset', default='test', metavar='SPLIT',
                           help='data subset to generate (train, valid, test)')
        group.add_argument('--num-shards', default=1, type=int, metavar='N',
                           help='shard generation over N shards')
        group.add_argument('--shard-id', default=0, type=int, metavar='ID',
                           help='id of the shard to generate (id < num_shards)')
    return group


def add_distributed_training_args(parser):
    group = parser.add_argument_group('Distributed training')
    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=8,
                       help='total number of NPUs across all nodes (default: all visible NPUs)')
    group.add_argument('--distributed-rank', default=0, type=int,
                        help='node rank for distributed training')
    group.add_argument('--dist-backend', default='hccl', type=str,
                        help='distributed backend')
    group.add_argument('--addr', default='XX.XX.XX.XX', type=str,
                        help='master addr')
    group.add_argument('--port', default='1234', type=str,
                        help='ip port')
    group.add_argument('--device-id', default=0, type=int,
                        help='npu id to use.')
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')
    group.add_argument('--update-freq', default='1', metavar='N',
                       help='update parameters every N_i batches, when in epoch i')

    # Optimizer definitions can be found under optim/
    group.add_argument('--optimizer', default='nag', metavar='OPT',
                       choices=OPTIMIZER_REGISTRY.keys(),
                       help='optimizer: {} (default: nag)'.format(', '.join(OPTIMIZER_REGISTRY.keys())))
    group.add_argument('--lr', '--learning-rate', default='0.25', metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--momentum', default=0.99, type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')

    # Learning rate schedulers can be found under optim/lr_scheduler/
    group.add_argument('--lr-scheduler', default='reduce_lr_on_plateau',
                       help='learning rate scheduler: {} (default: reduce_lr_on_plateau)'.format(
                           ', '.join(LR_SCHEDULER_REGISTRY.keys())))
    group.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                       help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
    group.add_argument('--min-lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                       help='minimum loss scale (for FP16 training)')

    # Criterion args
    parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                        help='epsilon for label smoothing, 0 means no label smoothing')

    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename in save-dir from which to load checkpoint')
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep last N checkpoints saved with --save-interval-updates')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                       help='validate every N epochs')
    return group



def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')

    # Model definitions can be found under models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    group.add_argument(
        '--arch', '-a', default='fconv', metavar='ARCH', required=True,
        choices=ARCH_MODEL_REGISTRY.keys(),
        help='model architecture: {} (default: fconv)'.format(
            ', '.join(ARCH_MODEL_REGISTRY.keys())),
    )

    # Criterion definitions can be found under criterions/
    group.add_argument(
        '--criterion', default='cross_entropy', metavar='CRIT',
        choices=CRITERION_REGISTRY.keys(),
        help='training criterion: {} (default: cross_entropy)'.format(
            ', '.join(CRITERION_REGISTRY.keys())),
    )

    return group