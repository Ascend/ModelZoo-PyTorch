# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist, load_state_dict

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--data_root',
        help='the path of dataset',
        type=str)
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='whether fine-tune model, change class num + 1')
    parser.add_argument('--total_epochs', type=int, default=12, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opt-level',
        choices=['O0', 'O1', 'O2'],
        default=None,
        help='apex opt-level')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.data.train.ann_file = cfg.data_root + 'annotations/instances_train2017.json'
        cfg.data.train.img_prefix = cfg.data_root + 'train2017/'
        cfg.data.val.ann_file = cfg.data_root + 'annotations/instances_val2017.json'
        cfg.data.val.img_prefix = cfg.data_root + 'val2017/'

    cfg.total_epochs = args.total_epochs
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.opt_level = args.opt_level
    if args.resume_from is not None and not args.fine_tune:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        # cfg.gpu_ids = args.gpu_ids
        # print('args.gpu_ids', args.gpu_ids[0])
        torch.npu.set_device(args.gpu_ids[0])
    cfg.gpus = args.gpus
    print('args.gpus', args.gpus)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # print('distributed gpu_ids', args.gpu_ids[0])
        # os.environ['NPUID'] = str(args.gpu_ids[0])
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)

    # fine-tune the model, default modify class + 1
    if args.fine_tune is not None and args.resume_from is not None:
        cfg.model.bbox_head.num_classes += 1

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]

    # fine-tune the model, default modify class + 1
    if args.fine_tune is not None and args.resume_from is not None:
        state_dict = torch.load(args.resume_from)['state_dict']
        load_state_dict(model, state_dict)

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp)


if __name__ == '__main__':
    main()