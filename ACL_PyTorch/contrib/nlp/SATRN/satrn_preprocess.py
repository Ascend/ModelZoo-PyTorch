# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
from pathlib import Path
import tqdm

import torch
import mmcv
from mmcv import Config
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
                        description='data preprocess.')
    parser.add_argument('--cfg-path', type=str, required=True, 
                        help='Test config file path.')
    parser.add_argument('--save-dir', type=str, required=True, 
                        help='a directory to save binary files.')
    args = parser.parse_args()
    return args


def preprocess(config_path, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(config_path)

    default_loader_cfg = dict(seed=cfg.get('seed'), 
                              drop_last=False, dist=False)
    if torch.__version__ == 'parrots':
        default_loader_cfg.update(dict(prefetch_num=2, pin_memory=False))
    for k, v in cfg.data.items():
        if k in ['train', 'val', 'test', 'train_dataloader', 
                 'val_dataloader', 'test_dataloader']:
            continue
        default_loader_cfg[k] = v

    test_loader_cfg = {
        **default_loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=1)
    }

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    for data in tqdm.tqdm(data_loader):
        image_path = data['img_metas'][0].data[0][0]['filename']
        save_path = save_dir / f"{Path(image_path).stem}.bin"
        data['img'][0].numpy().tofile(save_path)


if __name__ == '__main__':

    os.environ['LOCAL_RANK'] = '0'
    args = parse_args()
    preprocess(args.cfg_path, args.save_dir)
