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
import os.path as osp
import mmcv
import numpy as np
from mmcv import Config
from mmaction import __version__
from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default='./mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py')
    parser.add_argument('--work-dir', default='./inputs',
                        help='the dir to save images')

    parser.add_argument('--data_root', type=str, default='./mmaction2/tools/data/ucf101/')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--name', default='out_bin', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    cfg.data.test.ann_file = os.path.join(args.data_root, cfg.data.test.ann_file[12:])
    cfg.data.test.data_prefix = os.path.join(args.data_root, cfg.data.test.data_prefix[12:])
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    
    file = os.path.join(args.data_root, 'ucf101_'+str(args.batch_size)+'.info')
    out_path = os.path.join(args.data_root, args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(file,'w') as lael_file:
        l = len(dataset)
        t = 0
        batch_size = args.batch_size
        blank = np.zeros((batch_size, 75, 3, 256, 256)).astype(np.float32)
        while (t+1) * batch_size <= l:
            for j in range(batch_size):
                data = dataset[t*batch_size+j]
                blank[j] = np.array(data['imgs']).astype(np.float32)
                label = np.array(data['label']).astype(np.uint16)
                lael_file.write(str(label)+'\n')
            path = os.path.join(out_path, str(t) + ".bin")
            blank.tofile(path)
            print(blank.shape)
            t += 1


if __name__ == '__main__':
    main()