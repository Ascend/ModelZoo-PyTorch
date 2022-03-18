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
import sys
import argparse
import os
sys.path.append('mmaction2')
from mmaction.datasets.builder import  build_dataset,build_dataloader
from mmcv import Config, DictAction
import parser

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--bts', help='batch_size')
    parser.add_argument('--output_path',
                default=f'./pre_base_clip1_bs16/',
                help='Directory path of binary output data')
    args = parser.parse_args()
    return args

def preprocess():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=int(args.bts),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)
    for i, value in enumerate(data_loader):
        print(value['imgs'].shape)
        video_ids = value['label'].numpy().tolist()
        if len(value['label'])==1:
            str_ids = str(video_ids[0])
        else:
            str_ids = '_'.join(str(i) for i in video_ids)
        batch_bin = value['imgs'].cpu().numpy()
        print('preprocessing ' + str(video_ids))

        save_dir = str(args.output_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/bin' + str(int(args.bts)*i) + '-' + str(int(args.bts)*(i+1)-1) + '_' + str_ids + '.bin'
        batch_bin.tofile(str(save_path))
        print( i, str(save_path), "save done!")

        print("-------------------next-----------------------------")


if __name__ == '__main__':
    preprocess()