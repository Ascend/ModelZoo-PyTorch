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
#!/usr/bin/env bash

import os

import argparse
import copy
import os.path as osp
import time
import warnings

import torch.distributed as dist

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger, register_module_hooks


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default='config/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py',
                        help='train config file path')
    parser.add_argument('--work-dir', default='./result',
                        help='the dir to save logs and models')
    parser.add_argument('--resume-from', default='.', help='the checkpoint file to resume from')
    parser.add_argument('--validate', action='store_true',
                        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--test-last', action='store_true',
                        help='whether to test the checkpoint after training')
    parser.add_argument('--test-best', action='store_true',
                        help='whether to test the best checkpoint (if applicable) after training')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', default=True,
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={},
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file. For example, '
                             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    if args.data_root != '.':
        cfg.data.train.ann_file = os.path.join(args.data_root, cfg.data.train.ann_file[9:])
        cfg.data.train.data_prefix = os.path.join(args.data_root, cfg.data.train.data_prefix[9:])

        cfg.data.val.ann_file = os.path.join(args.data_root, cfg.data.val.ann_file[9:])
        cfg.data.val.data_prefix = os.path.join(args.data_root, cfg.data.val.data_prefix[9:])

        cfg.data.test.ann_file = os.path.join(args.data_root, cfg.data.test.ann_file[9:])
        cfg.data.test.data_prefix = os.path.join(args.data_root, cfg.data.test.data_prefix[9:])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from != '.':
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        device = torch.device('npu:{}'.format(cfg.DEVICE_ID))
        torch.npu.set_device(device)
    else:
        distributed = True
        os.environ['NPUID'] = str(args.gpu_ids[0])
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatiable with val workflow,
        # we recommend you to use `--validate`
        assert not cfg.omnisource
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
