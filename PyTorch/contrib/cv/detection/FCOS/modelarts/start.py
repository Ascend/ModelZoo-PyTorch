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

import argparse
import copy
import os
import os.path as osp
import time
import warnings
import moxing as mox

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')

    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_npus = parser.add_mutually_exclusive_group()
    group_npus.add_argument(
        '--npus',
        type=int,
        help='number of npus to use '
        '(only applicable to non-distributed training)')
    group_npus.add_argument(
        '--npu-ids',
        type=int,
        nargs='+',
        help='ids of npus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    # add for apex
    parser.add_argument('--amp', default=False,
                        action='store_true', help='use amp to train the model')
    parser.add_argument('--loss-scale', default=32.0, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
                        help='loss scale using in amp, default -1 means dynamic')

    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg_options instead.')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # modelarts modification
    parser.add_argument('--pretrained',
                        default="/cache/pretrained/",
                        metavar='DIR',
                        help="path to pretrained model")

    parser.add_argument('--data_url',
                        metavar='DIR',
                        default='/cache/data_url',
                        help='path to dataset')

    parser.add_argument('--train_url',
                        default="/cache/training",
                        type=str,
                        help="setting dir of training output")

    parser.add_argument('--epochs', type=int, default=12, help='total epochs')

    parser.add_argument(
        '--load_from', help='the checkpoint file to load from')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg_options cannot be both '
            'specified, --options is deprecated in favor of --cfg_options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg_options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # modelarts modification

    real_path = '/cache/data_url/'
    model_path = "/cache/model"
    pretrained_path = "/cache/pretrained/"
    out_path = '/cache/training'
    if not os.path.exists(real_path):
        os.makedirs(real_path)

    mox.file.copy_parallel(args.data_url, real_path)
    print("---------------------------------------------------------")
    print("training data finish copy to %s." % real_path)
    print("---------------------------------------------------------")

    pres = args.pretrained.split("/")
    pre_name = pres[-1]

    pre_file = pretrained_path+pre_name
    print("---------------------------------------")
    print(args.pretrained)
    print("---------------------------------------")

    mox.file.copy(args.pretrained, pre_file)

    print("---------------------------------------------------------")
    print("training data finish copy to %s." % pre_file)
    print("---------------------------------------------------------")

    cfg.data_root = real_path
    cfg.data.train.ann_file = real_path+'annotations/instances_train2017.json'
    cfg.data.train.img_prefix = real_path+'train2017/'
    cfg.data.val.ann_file = real_path+'annotations/instances_val2017.json'
    cfg.data.val.img_prefix = real_path+'val2017/'
    cfg.data.test.ann_file = real_path+'annotations/instances_val2017.json'
    cfg.data.test.img_prefix = real_path+'val2017/'
    cfg.total_epochs = args.epochs
    cfg.model.pretrained = pre_file

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.train_url is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = out_path

    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.opt_level = args.opt_level  # add for apex
    cfg.loss_scale = args.loss_scale  # add for apex
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.load_from is not None:

        # modelarts Transfer learning

        mm = args.load_from.split("/")
        model_name = mm[-1]

        print("---------------------------------------")
        print(model_name)
        print("---------------------------------------")

        os.makedirs(model_path, exist_ok=True)
        mox.file.copy(args.load_from, os.path.join(model_path, model_name))
        cfg.load_from = os.path.join(model_path, model_name)

    if args.npu_ids is not None:
        cfg.npu_ids = args.npu_ids
        torch.npu.set_device(cfg.npu_ids[0])
    else:
        cfg.npu_ids = range(1) if args.npus is None else range(args.npus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        os.environ['NPUID'] = str(args.npu_ids[0])  # add NPUID
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.npu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
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
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    # modelarts modification

    mox.file.copy_parallel(out_path, args.train_url)
    print("---------------------------------------------------------")
    print("output data finish copy to %s." % args.train_url)
    print("---------------------------------------------------------")


if __name__ == '__main__':
    main()
