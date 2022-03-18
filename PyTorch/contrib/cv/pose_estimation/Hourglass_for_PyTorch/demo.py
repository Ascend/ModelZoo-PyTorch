# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument(
        '--config', default='mmpose-master/configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py', help='test config file path')
    parser.add_argument('--checkpoint', default='mmpose-master/work_dirs/hourglass52_mpii_384x384/latest.pth', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        default='PCKh',
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
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


def infer_out():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build the dataloader
    dataset = build_dataset(cfg.data.demo, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.npu(), device_ids=[0])

    # for inference
    outputs = single_gpu_test(model, data_loader)

    # post-processing
    img_prefix_path = cfg.data.demo["img_prefix"]
    img_name_original = str(outputs[0][2])
    img_name_original = img_name_original.replace(',', '')
    img_name_original = img_name_original.replace('[', '')
    img_name_original = img_name_original.replace(']', '')
    img_name_original = img_name_original.replace("'", '')
    img_name_original = img_name_original.replace(" ", '')

    img_name_only = osp.basename(img_name_original)
    img_path = osp.join(os.getcwd(), img_name_original)
    img_write_path = osp.join(os.getcwd(), img_prefix_path)
    img_write_path = osp.join(img_write_path, "infer_"+img_name_only)
    
    img_pre = outputs[0][0][0]
    img = cv2.imread(img_path)
    skeleton_line = [[0, 1], [1, 2], [2, 12], [12, 11], [11, 10], [12, 7], [7, 8], [
        8, 9], [7, 6], [7, 13], [13, 14], [14, 15], [13, 3], [3, 6], [6, 2], [3, 4], [4, 5]]
    
    # draw the key_point
    for coordirate in list(img_pre):
        x = coordirate[0]
        y = coordirate[1]
        score = coordirate[2]
        if score > 0.3:
            cv2.circle(img, (x, y), 7, (0, 0, 255), -1)
    # draw skeleton_line
    for line_points in skeleton_line:
        score1 = img_pre[line_points[0]][2]
        score2 = img_pre[line_points[1]][2]
        if(score1 > 0.3 and score2 > 0.3):
            x1 = img_pre[line_points[0]][0]
            y1 = img_pre[line_points[0]][1]
            x2 = img_pre[line_points[1]][0]
            y2 = img_pre[line_points[1]][1]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(img_write_path, img)


if __name__ == '__main__':
    infer_out()
