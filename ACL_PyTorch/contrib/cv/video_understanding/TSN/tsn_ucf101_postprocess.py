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
import numpy as np
from tqdm import tqdm
import mmcv
from mmcv import Config
from mmaction import __version__
from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset UCF101 Postprocessing')
    parser.add_argument('--config', default='./mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py')
    parser.add_argument('--work-dir', default='./inputs',
                        help='the dir to save images')

    parser.add_argument('--data_root', required=True, type=str, default='./mmaction2/tools/data/ucf101/')
    parser.add_argument('--result_path', required=True, type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    cfg.data.test.ann_file = os.path.join(args.data_root, cfg.data.test.ann_file[12:])
    cfg.data.test.data_prefix = os.path.join(args.data_root, cfg.data.test.data_prefix[12:])
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    acc = 0
    files = os.listdir(args.result_path)
    for file in tqdm(files):
        with open(os.path.join(args.result_path, file)) as f:
            lines = f.readlines()
            lines = list(map(lambda x:x.strip().split(), lines))
            lines = np.array([float(lines[0][n]) for n in range(101)]).argmax(0)
            
        i = int(file.split('_')[0])
        label = np.array(dataset[i]['label']).astype(np.uint16)
        acc += int(lines == label)

    print(acc / len(files))


if __name__ == '__main__':
    main()
