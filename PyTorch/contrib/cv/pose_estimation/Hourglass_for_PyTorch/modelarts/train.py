# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')

    parser.add_argument('--ckpt_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--total_epochs', type=int, default=1)
    parser.add_argument('--work_dir', type=str, default="./output") # output file to save log and models
    parser.add_argument('--data_root', type=str, default="../data/mpii") # dataset root file
    parser.add_argument('--config', type=str,
                        default="../mmpose-master/configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py")

    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true')
    
    group_npus = parser.add_mutually_exclusive_group()
    group_npus.add_argument('--npus', type=int, default=1, help='number of gpus to use')
    group_npus.add_argument('--npu_ids', type=int, nargs='+', help='ids of gpus to use')

    parser.add_argument('--autoscale-lr',
                        action='store_true',
                        help='automatically scale lr with the number of gpus')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch', help='job launcher')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg.checkpoint_config.interval = args.ckpt_interval
    cfg.evaluation.interval = args.eval_interval
    cfg.optimizer.lr = args.lr
    cfg.lr_config.warmup_iters = args.warmup_iters
    cfg.lr_config.warmup_ratio = args.warmup_ratio
    cfg.total_epochs = args.total_epochs
    cfg.work_dir = args.work_dir
    cfg.data.train.ann_file = f'{args.data_root}/annotations/mpii_train.json'
    cfg.data.train.img_prefix = f'{args.data_root}/images/'
    cfg.data.val.ann_file = f'{args.data_root}/annotations/mpii_val.json'
    cfg.data.val.img_prefix = f'{args.data_root}/images/'
    cfg.data.test.ann_file = f'{args.data_root}/annotations/mpii_test.json'
    cfg.data.test.img_prefix = f'{args.data_root}/images/'

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.num_of_npus = args.npus
    if args.npu_ids is not None:
        cfg.npu_ids = args.npu_ids
        torch.npu.set_device(cfg.npu_ids[0])
    else:
        cfg.npu_ids = range(1) if args.npus is None else range(args.npus)

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.npu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if (args.seed != 0):
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed
    else:
        cfg.seed = None
        meta['seed'] = None        

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()