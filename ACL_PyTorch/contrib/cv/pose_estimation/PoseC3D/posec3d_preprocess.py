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

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from mmcv import Config
import torch.nn.functional as F
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset HMDB51 Preprocessing')
    parser.add_argument('--config',
                        default='./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py',
                        help='config file path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_worker', default=8, type=int, help='Number of workers for inference')
    parser.add_argument('--data_root', default='/opt/npu/hmdb51/rawframes/', type=str)
    parser.add_argument('--ann_file', default='/opt/npu/hmdb51/hmdb51.pkl', type=str)
    parser.add_argument('--name', default='./prep_hmdb51_bs1', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.data_root

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=args.num_worker,
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    root_path = os.path.dirname(args.ann_file)
    out_path = args.name
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    info_file = open(os.path.join(root_path, 'hmdb51.info'), 'w')

    pbar = tqdm(data_loader)
    i = 0
    for data in pbar:
        pbar.set_description('Preprocessing ')
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            info_file.write(str(args.batch_size*i+batch) + ' ' + str(l))
            info_file.write('\n')

        if imgs.shape[0] != args.batch_size:
            imgs = F.pad(imgs, (0,0,0,0,0,0,0,0,0,args.batch_size-imgs.shape[0]))

        bin_info = imgs.cpu().numpy()
        bin_info.tofile(out_path + '/' + str(i) + '.bin')
        i += 1

if __name__ == '__main__':
    main()