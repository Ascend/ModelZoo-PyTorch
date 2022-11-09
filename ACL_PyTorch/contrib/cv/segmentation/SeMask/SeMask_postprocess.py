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
"""Postprocess module"""

import argparse
import os
import json
import numpy as np
import torch

import mmcv
from mmcv.runner import init_dist
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    """Postprocess arguments"""
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--input_dir', type=str,
                        default='./infer/preprocess_result/leftImg8bit', help='input data dictionary')
    parser.add_argument('--result_dir', type=str,
                        default='./output', help='result data dictionary')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--data_root',
        type=str,
        default="./datasets/cityscapes/",
        help='data file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
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
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
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

    filesinfo = {}
    with open(os.path.join(args.result_dir, 'sumary.json'), encoding='utf-8') as f:
        sumary = json.load(f)
    for i, info in sumary['filesinfo'].items():
        input_name = info['infiles'][0].split('/')[-1].split('.')[0]
        output_name = info['outfiles'][0].split('/')[-1]
        filesinfo[input_name] = output_name

    output_path = []
    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data[0]
        filename = img_metas[0]['filename']
        filename = filename.split('/')[-1].split('.')[0]
        result_path = os.path.join(args.result_dir, filesinfo[filename])
        output_path.append(result_path)

    outputs = []
    for i, _ in enumerate(output_path):
        output = np.fromfile(output_path[i], np.int64)
        output = np.array(output).reshape(1024, 2048)  # 转换为对应的shape
        outputs.append(output)

    kwargs = {} if args.eval_options is None else args.eval_options
    dataset.evaluate(outputs, 'mIoU', **kwargs)
