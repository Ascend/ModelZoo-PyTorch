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
import argparse

from tqdm import tqdm
import torch.nn.functional as F
from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset Kinetics400 Preprocessing')
    parser.add_argument('--config', type=str, 
                        help='config file path')
    parser.add_argument('--num_worker', type=int, default=8,
                        help='Number of workers for inference')
    parser.add_argument('--video_dir', type=str,
                        default='mmaction2/data/kinetics400/videos_val',
                        help='path to test videos')
    parser.add_argument('--ann_file', type=str, 
                        default='mmaction2/data/kinetics400/kinetics400_val_list_videos.txt',
                        help='path to video list file')
    parser.add_argument('--save_dir', type=str,
                        help='path to save preprocess result')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    batch_size = 1

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.video_dir

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(videos_per_gpu=1,
                              workers_per_gpu=args.num_worker,
                              dist=False,
                              shuffle=False)
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    bin_dir = os.path.join(args.save_dir, 'bin')
    if not os.path.isdir(bin_dir):
        os.mkdir(bin_dir)
    info_path = os.path.join(args.save_dir, 'kinetics400.info')
    file = open(info_path, 'w')

    for i, data in enumerate(tqdm(data_loader)):
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            file.write(str(batch_size * i + batch) + ' ' + str(l))
            file.write('\n')

        if imgs.shape[0] != batch_size:
            print(f'Num of data in the last batch is {imgs.shape[0]}')
            imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                batch_size - imgs.shape[0]))

        bin = imgs.cpu().numpy()
        bin.tofile(os.path.join(bin_dir, str(i) + '.bin'))


if __name__ == '__main__':
    main()
