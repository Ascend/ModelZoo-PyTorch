# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    
    # NPU
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--loss_scale', default=32.0, type=float)
    parser.add_argument('--opt_level', default='O1', type=str, choices=['O0', 'O1', 'O2']) 
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## for modelArts training
    ## ===========================
    parser.add_argument('--data-dir', help='the path of data set')
    parser.add_argument('--pretrained', default='torchvision://resnet50', 
        help='the checkpoint file to load for pretrain')
    parser.add_argument('--load-from', help='the checkpoint file to load from')

    parser.add_argument('--max_epochs', type=int, default=12, help='max training epoch')
    # parser.add_argument('--num_classes', type=int, default=80, help='the num of dataser classes')
    # parser.add_argument('--classes', type=str, help='all classes')

    parser.add_argument('--train_ann_file', type=str, default='annotations/instances_train2017.json' ,help='train annotations')
    parser.add_argument('--train_img_prefix', type=str, default='train2017/' ,help='train img prefix')
    parser.add_argument('--val_ann_file', type=str, default='annotations/instances_val2017.json' ,help='validate annotations')
    parser.add_argument('--val_img_prefix', type=str, default='val2017/' ,help='val img prefix')
    ## ===========================

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    ## for modelArts training
    ## ================================ 
    local_data_path = '/cache/data_url/'            # data set url
    local_pretrained_path = '/cache/checkpoint/'    # pretrained pth path
    local_output_path = '/cache/output'             # output result(pth, log)
    local_model_path = '/cache/model'               # for transfer learning
    # 1. copy dataset to modelarts local cache
    os.makedirs(local_data_path, exist_ok=True)
    # args.data_dir -> local_data_path
    mox.file.copy_parallel(args.data_dir, local_data_path)
    cfg.data_root = local_data_path
    cfg.data.train.ann_file = local_data_path + args.train_ann_file
    cfg.data.train.img_prefix = local_data_path + args.train_img_prefix
    cfg.data.val.ann_file = local_data_path + args.val_ann_file
    cfg.data.val.img_prefix = local_data_path + args.val_img_prefix
    cfg.data.test.ann_file = local_data_path + args.val_ann_file
    cfg.data.test.img_prefix = local_data_path + args.val_img_prefix

    # if args.num_classes and args.classes:
    #     class_tuple = tuple(args.classes.split(','))
    #     if len(class_tuple) != args.num_classes:
    #         raise ValueError(
    #         '--classes and --num_classes dont match')
    #     # set config
    #     cfg.classes = class_tuple
    #     cfg.data.train.classes = class_tuple
    #     cfg.data.val.classes = class_tuple
    #     cfg.data.test.classes = class_tuple
    # cfg.model.bbox_head.num_classes = args.num_classes

    # 2. copy pretrained pth to modelarts local cache
    os.makedirs(local_pretrained_path, exist_ok=True)
    pth_name = args.pretrained.split("/")[-1]
    local_pretrained_pth = local_pretrained_path + pth_name

    mox.file.copy(args.pretrained, local_pretrained_pth)
    cfg.model.pretrained = local_pretrained_pth

    # 3. copy load pth to modelarts local cache for model transfering
    if args.load_from is not None:
        model_name = args.load_from.split("/")[-1]
        local_model_pth = local_model_path + model_name

        os.makedirs(local_model_path, exist_ok=True)
        mox.file.copy(args.load_from, local_model_pth)
        cfg.load_from = local_model_pth

    # 4. modify max epoch config
    cfg.runner.max_epochs = args.max_epochs


    ## ================================

    cfg.opt_level = args.opt_level
    cfg.loss_scale = args.loss_scale

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
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        # for modelArts training
        # save the output locally
        cfg.work_dir = local_output_path
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # NPU
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
        # NPU
        # retinanet
        logger.info("===================使用NPU====================")
        torch.npu.set_device(cfg.gpu_ids[0])
        logger.info(str(cfg.gpu_ids[0]))
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # NPU
        # retinanet
        # init_dist(args.launcher, **cfg.dist_params)
        # os.environ['NPUID'] = str(args.gpu_ids[0])
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

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
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

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

    ## for modelArts training
    ## ===========================
    # copy output result to obs
    mox.file.copy_parallel(local_output_path, args.work_dir)
    ## ===========================


if __name__ == '__main__':
    main()
