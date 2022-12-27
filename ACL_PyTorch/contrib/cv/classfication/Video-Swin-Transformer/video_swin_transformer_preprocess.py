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

import argparse
import os
import os.path as osp
import warnings
import sys
import numpy as np
import torch.nn.functional as F
import mmcv
import torch
from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--save_path',
        default=None,
        help='the path of om result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    dataset_type = cfg.data.test.type

    cfg.data.test.test_mode = True


    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)
    
    print("preprocess begin")
    save_path = args.save_path
    length = len(dataset)
    print("the total dataset:",len(dataset))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)
    for i , data in enumerate(data_loader,1):
      batch_bin = data['imgs'].cpu().numpy()
      batch_bin.tofile(osp.join(save_path,str(i)+'.bin'))
      print("\r", end="")
      print("processing: {}%: ".format((i * 100) // length), "▓" * (((i * 100) // length) // 2), end="")
      sys.stdout.flush()
    print("\n")
    print("preprocess finished")


if __name__ == '__main__':
    main()
