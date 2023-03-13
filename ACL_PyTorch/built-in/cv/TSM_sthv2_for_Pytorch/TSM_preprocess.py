"""
Copyright 2023 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from mmcv import Config
import sys
sys.path.append('mmaction2')
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    """
    input argument receiving function
    :return: input arguments
    """
    parser = argparse.ArgumentParser(
        description='Dataset Sthv2 Preprocessing')
    parser.add_argument('--config',
                        default='./mmaction2/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py',
                        help='config file path')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for inference')
    parser.add_argument('--num_worker', default=8, type=int,
                        help='Number of workers for inference')
    parser.add_argument('--data_root',
                        default='./mmaction2/data/sthv2/rawframes/',
                        type=str)
    parser.add_argument('--ann_file',
                        default='./mmaction2/data/sthv2/sthv2_val_list_rawframes.txt',
                        type=str)
    parser.add_argument('--output_dir', default='out_bin', type=str)

    args = parser.parse_args()

    return args


def main():
    """
    main function
    :return: None
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.data_root

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(**cfg.data.get('test_dataloader', {}))
    dataloader_setting['videos_per_gpu'] = args.batch_size
    dataloader_setting['workers_per_gpu'] = args.num_worker
    dataloader_setting['dist'] = False
    dataloader_setting['shuffle'] = False

    data_loader = build_dataloader(dataset, **dataloader_setting)

    root_path = './sthv2'
    out_path = os.path.join(root_path, args.output_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(root_path, 'sthv2.info'), 'w') as file_:
        for i, data in tqdm(enumerate(data_loader)):
            imgs = data['imgs']
            label = data['label']

            for batch in range(imgs.shape[0]):
                file_.write(
                    f'{args.batch_size * i + batch} {label.cpu().numpy()[batch]}')
                file_.write('\n')

            if imgs.shape[0] != args.batch_size:
                imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    args.batch_size - imgs.shape[0]))

            imgs.cpu().numpy().tofile(os.path.join(out_path, f'{i}.bin'))


if __name__ == '__main__':
    main()
