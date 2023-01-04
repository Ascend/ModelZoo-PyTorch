# Copyright 2022 Huawei Technologies Co., Ltd
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


import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
                        description='Dataset HMDB51 Preprocessing')
    parser.add_argument('--config_file',
                        default='./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py',
                        help='path to config file.')
    parser.add_argument('--frame_dir', type=str,
                        help='directory of raw frames.')
    parser.add_argument('--ann_file', type=str,
                        help='path to annotation file.')
    parser.add_argument('--output_dir', type=str, default='prep_dataset',
                        help='directory to save results of preprocess.')
    parser.add_argument('--num_worker', default=8, type=int, 
                        help='number of workers for load data.')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    # load config.
    cfg = Config.fromfile(args.config_file)

    # build the dataloader
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.frame_dir
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=1,
        workers_per_gpu=args.num_worker,
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, 
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # prepare
    bin_dir = osp.join(args.output_dir, 'bin')
    if not osp.isdir(bin_dir):
        os.makedirs(bin_dir)
    info_file = open(osp.join(args.output_dir, 'hmdb51_label.txt'), 'w')

    # save data with binary files.
    for i, data in enumerate(tqdm(data_loader, desc='Preprocessing')):
        bin_name = f'{i:0>5d}'
        label = data['label'].cpu().numpy()[0]
        info_file.write(f'{bin_name} {label}\n')
        bin_path = osp.join(bin_dir, bin_name + '.bin')
        if osp.isfile(bin_path) and osp.getsize(bin_path) == 204718080:
            continue
        img = data['imgs'].cpu().numpy()
        img.tofile(bin_path)
    info_file.close()


if __name__ == '__main__':
    main()