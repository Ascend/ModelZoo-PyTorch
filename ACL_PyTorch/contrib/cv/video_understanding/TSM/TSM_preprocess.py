"""
Copyright 2020 Huawei Technologies Co., Ltd

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

from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    """
    input argument receiving function
    :return: input arguments
    """
    parser = argparse.ArgumentParser(description='Dataset UCF101 Preprocessing')
    parser.add_argument('--config',
                        default='./mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py',
                        help='config file path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_worker', default=8, type=int, help='Number of workers for inference')
    parser.add_argument('--data_root', default='./mmaction2/tools/data/ucf101/rawframes/', type=str)
    parser.add_argument('--ann_file', default='./mmaction2/tools/data/ucf101/ucf101_val_split_1_rawframes.txt',
                        type=str)
    parser.add_argument('--name', default='out_bin', type=str)

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

    root_path = './ucf101'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    out_path = os.path.join(root_path, args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    file_ = open(os.path.join(root_path, 'ucf101.info'), 'w')

    for i, data in enumerate(data_loader):
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            file_.write('{0} {1}'.format(args.batch_size * i + batch,l))
            file_.write('\n')

        if imgs.shape[0] != args.batch_size:
            imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, 0, 0, args.batch_size - imgs.shape[0]))
            print(imgs.shape[0] != args.batch_size)

        bin_ = imgs.cpu().numpy()
        bin_.tofile(out_path + '/' + str(i) + '.bin')
    file_.close()
    print('Finish preprocessing for batch size {}'.format(args.batch_size))


if __name__ == '__main__':
    main()
