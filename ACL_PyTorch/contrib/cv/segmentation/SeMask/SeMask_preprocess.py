# Copyright 2021 Huawei Technologies Co., Ltd
#
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
# ============================================================================
"""Preprocess module"""

from __future__ import print_function
import os
import argparse
import numpy as np
import torch

import mmcv
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    """Preprocess arguments"""
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--data_root',
        type=str,
        default="./datasets/cityscapes/",
        help='data file path')
    parser.add_argument('--save_path', type=str,
                        default='./preprocess_result', help='input data save path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    cfg.data.test.data_root = args.data_root
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    print("=" * 20, 'start pretreatment', "=" * 20)
    save_path = os.path.join(args.save_path + '/leftImg8bit')
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path + '/leftImg8bit')):
        os.mkdir(os.path.join(args.save_path + '/leftImg8bit'))
    print(f"images_bin stored in ${os.path.join(args.save_path + '/leftImg8bit')}")
    for i, data in enumerate(data_loader):
        imgs = data['img'][0]
        imgs = np.array(imgs).astype(np.float32)
        img_metas = data['img_metas'][0].data[0]
        filename = img_metas[0]['filename']
        imgs.tofile(os.path.join(save_path, filename.split('/')[-1].split('.')[0] + ".bin"))
    print("=" * 20, 'end pretreeatmen', "=" * 20)
