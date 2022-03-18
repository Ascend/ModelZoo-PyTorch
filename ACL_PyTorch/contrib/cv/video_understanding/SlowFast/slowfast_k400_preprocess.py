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
import os
import argparse
import torch.nn.functional as F

from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset Kinetics400 Preprocessing')
    parser.add_argument(
        '--config',
        default=
        'mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py',
        help='config file path')
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='Batch size for inference')
    parser.add_argument('--num_worker',
                        default=8,
                        type=int,
                        help='Number of workers for inference')
    parser.add_argument('--data_root',
                        default='mmaction2/data/kinetics400/videos_val',
                        type=str)
    parser.add_argument(
        '--ann_file',
        default='mmaction2/data/kinetics400/kinetics400_val_list_videos.txt',
        type=str)
    parser.add_argument('--name', default='out_bin', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.data_root

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(videos_per_gpu=args.batch_size,
                              workers_per_gpu=args.num_worker,
                              dist=False,
                              shuffle=False)
    data_loader = build_dataloader(dataset, **dataloader_setting)

    root_path = os.path.dirname(args.ann_file)
    out_path = os.path.join(root_path, args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    file = open(os.path.join(root_path, 'k400.info'), 'w')

    for i, data in enumerate(data_loader):
        print('Preprocessing video {}/{}'.format(i + 1, len(data_loader)))
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            file.write(str(args.batch_size * i + batch) + ' ' + str(l))
            file.write('\n')

        if imgs.shape[0] != args.batch_size:
            print(f'Num of data in the last batch is {imgs.shape[0]}')
            imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                args.batch_size - imgs.shape[0]))

        bin = imgs.cpu().numpy()
        bin.tofile(out_path + '/' + str(i) + '.bin')


if __name__ == '__main__':
    main()
