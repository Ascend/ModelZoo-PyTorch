# Copyright 2021 Huawei Technologies Co., Ltd

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import torch
import os                       # for getenv()

from argparse import ArgumentParser

import random

# adds mutually exclusive "--name" and "--no-name" command line arguments, with
# the result stored in a variable named "name" (with any dashes in "name"
# replaced by underscores)
# inspired by https://stackoverflow.com/a/31347222/2209313
def add_bool_arg(group, name, default=False, help=''):
    subgroup = group.add_mutually_exclusive_group(required=False)
    name_with_underscore = name.replace('-', '_').replace(' ', '_')

    truehelp = help
    falsehelp = help
    if help != '':
        falsehelp = 'do not ' + falsehelp
    if default is True:
        if truehelp != '':
            truehelp = truehelp + ' '
        truehelp = truehelp + '(default)'
    else:
        if falsehelp != '':
            falsehelp = falsehelp + ' '
        falsehelp = falsehelp + '(default)'

    subgroup.add_argument('--' + name, dest=name_with_underscore, action='store_true', help=truehelp)
    subgroup.add_argument('--no-' + name, dest=name_with_underscore, action='store_false', help=falsehelp)
    group.set_defaults(**{name_with_underscore:default})

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")

    data_group = parser.add_argument_group('data', 'data-related options')
    # Data-related
    data_group.add_argument('--data', '-d', type=str, default='/home/ssd_zjy/coco/',
                        help='path to test and training data files')
    data_group.add_argument('--meta_files_path', type=str, default=None,
                        help='path to COCO meta files')
    data_group.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    data_group.add_argument('--eval-batch-size', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    # input pipeline stuff
    add_bool_arg(data_group, 'dali', default=False)  # --dali (default) and --no-dali
    data_group.add_argument('--fake-input', action='store_true',
                        help='run input pipeline with fake data (avoid all i/o and work except on very first call)')
    data_group.add_argument('--input-batch-multiplier', type=int, default=1,
                        help='run input pipeline at batch size <n> times larger than that given in --batch-size')
    data_group.add_argument('--dali-sync', action='store_true',
                        help='run dali in synchronous mode instead of the (default) asynchronous')
    data_group.add_argument('--dali-cache', type=int, default=0,
                        help="cache size (in GB) for Dali's nvjpeg caching")
    data_group.add_argument('--use-nvjpeg', default=False)
    data_group.add_argument('--use-roi-decode', default=False,
                            help="DEPRECATED: Dali input pipeline uses roi decode if and only if --dali-cache is not set" )
    data_group.add_argument('--device_id', type=int, default=0)
    data_group.add_argument('--tag', type=str, default=None)
    data_group.add_argument('--loss_scale', type=int, default=32)
    # model-related
    model_group = parser.add_argument_group('model', 'Model-related options')
    model_group.add_argument('--model-path', type=str, default='./vgg16n.pth')
    model_group.add_argument('--backbone', type=str, choices=['vgg16', 'vgg16bn', 'resnet18', 'resnet34', 'resnet50'], default='resnet34')
    model_group.add_argument('--num-workers', type=int, default=4)
    model_group.add_argument('--use-fp16', action='store_true')
    model_group.add_argument('--print-interval', type=int, default=20)
    model_group.add_argument('--jit', action='store_true')
    model_group.add_argument('--nhwc', default=False)
    model_group.add_argument('--pad-input', default=False)
    model_group.add_argument('--num-classes', type=int, default=81)
    model_group.add_argument('--input-size', type=int, default=300)

    # Solver-related
    solver_group = parser.add_argument_group('solver', 'Solver-related options')
    solver_group.add_argument('--epochs', '-e', type=int, default=90,
                        help='number of epochs for training')
    add_bool_arg(solver_group, 'allreduce-running-stats', default=True,
                 help='allreduce batch norm running stats before evaluation')
    solver_group.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    solver_group.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    solver_group.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    solver_group.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    add_bool_arg(solver_group, 'save', default=True,
                        help='save model checkpoints')
    solver_group.add_argument('--evaluation', nargs='*', type=int,
                              default=[60, 65, 70, 75, 80, 85],
                              help='epochs at which to evaluate')
    solver_group.add_argument('--warmup', type=int, default=None)
    solver_group.add_argument('--warmup-factor', type=int, default=1,
                        help='mlperf rule parameter for controlling warmup curve')
    solver_group.add_argument('--lr', type=float, default=2.68e-3)
    solver_group.add_argument('--wd', type=float, default=5e-4)
    solver_group.add_argument('--lr-decay-factor', type=float, default=0.1,
                              help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=[45,58,84],
                        help='epochs at which learning rate decays. default is 44,55.')
    solver_group.add_argument('--delay-allreduce', action='store_true')
    solver_group.add_argument('--opt-loss', action='store_true', help='deprecated option, does nothing (loss is always optimized)')
    solver_group.add_argument('--bn-group', type=int, default=1, choices=[1, 2, 4, 8], help='Group of processes to collaborate on BatchNorm ops')

    # Profiling
    profiling_group = parser.add_argument_group('profiling', 'Profiling options')
    profiling_group.add_argument('--profile', type=int, default=None,
                        help='iteration at which to early terminate')
    profiling_group.add_argument('--profile-start', type=int, default=None,
                        help='iteration at which to turn on cuda and/or pytorch nvtx profiling')
    profiling_group.add_argument('--profile-nvtx', action='store_true',
                        help='turn on pytorch nvtx annotations in addition to cuda profiling')
    profiling_group.add_argument('--profile-gc-off', action='store_true',
                                 help='call gc.disable() (useful for eliminating gc noise while profiling)')
    profiling_group.add_argument('--profile-cudnn-get', action='store_true',
                                 help='use cudnnGet() rather than cudnnFind() to eliminate a possible source of perf non-determinism')
    profiling_group.add_argument('--profile-fake-optim', action='store_true',
                                 help='turn off optimizer to get more accurate timing of the rest of the training pipe')

    # Distributed stuff
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int,
			help='Used for multi-process training. Can either be manually set ' +
			'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()

# make sure that arguments are all self-consistent
def validate_arguments(args):
    # nhwc can only be used with fp16
    if args.nhwc:
        assert(args.use_fp16)

    # input padding can only be used with NHWC
    if args.pad_input:
        assert(args.nhwc)

    # no dali can only be used with NCHW and no padding
    if not args.dali:
        assert(not args.nhwc)
        assert(not args.pad_input)
        assert(not args.use_nvjpeg)
        assert(not args.dali_cache)
        assert(not args.use_roi_decode)

    if args.use_roi_decode:
        assert(args.dali_cache<=0) # roi decode also crops every epoch, so can't cache

    if args.dali_cache>0:
        assert(args.use_nvjpeg)

    if args.jit:
        assert(args.nhwc) #jit can not be applied with apex::syncbn used for non-nhwc

    return

# Check that the run is valid for specified group BN arg
def validate_group_bn(bn_groups):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    # Can't have larger group than ranks
    assert(bn_groups <= world_size)

    # must have only complete groups
    assert(world_size % bn_groups == 0)

