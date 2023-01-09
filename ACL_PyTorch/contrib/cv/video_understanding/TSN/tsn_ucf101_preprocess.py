"""
Copyright 2022 Huawei Technologies Co., Ltd

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

from tqdm import tqdm
import mmcv
import numpy as np
from mmcv import Config
from mmaction import __version__
from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset UCF101 Preprocessing')
    parser.add_argument('--config', default='./mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py')
    parser.add_argument('--work-dir', default='./inputs',
                        help='the dir to save images')

    parser.add_argument('--data_root', required=True, type=str, default='./mmaction2/tools/data/ucf101/')
    parser.add_argument('--save_dir', required=True, default='prep_bin', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    cfg.data.test.ann_file = os.path.join(args.data_root, cfg.data.test.ann_file[12:])
    cfg.data.test.data_prefix = os.path.join(args.data_root, cfg.data.test.data_prefix[12:])
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        arr = np.array(data['imgs']).astype(np.float32)
        path = os.path.join(save_dir, str(i) + ".bin")
        arr.tofile(path)


if __name__ == '__main__':
    main()