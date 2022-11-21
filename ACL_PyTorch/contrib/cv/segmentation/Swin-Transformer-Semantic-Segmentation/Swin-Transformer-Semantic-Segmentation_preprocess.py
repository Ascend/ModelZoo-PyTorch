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
import tqdm
from pathlib import Path


import numpy as np
import mmcv
from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
                        description='process images and save to binary files.')
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file.')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='a directory to save output binary files.')
    args = parser.parse_args()
    return args


def slide_crop(img, img_meta, model_cfg, save_dir):
    """slide window and crop image"""
    ori_shape = img_meta[0]['ori_shape']
    stem = Path(img_meta[0]['ori_filename']).stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    h_stride, w_stride = model_cfg.test_cfg.stride
    h_crop, w_crop = model_cfg.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    cnt = 0
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            
            cnt += 1
            msg1 = f'{x1},{y1},{x2},{y2}'
            msg2 = f'{ori_shape[0]},{ori_shape[1]},{ori_shape[2]}'
            msg3 = f'{batch_size},{h_img},{w_img}'
            bin_path = save_dir/f'{stem}-{"-".join([msg1, msg2, msg3])}.bin'
            crop_img.numpy().astype(np.float32).tofile(bin_path)

    return cnt


def preprocess(config_path, save_dir):
    """process original images and save to binary files"""
    
    cfg = mmcv.Config.fromfile(config_path)
    cfg.merge_from_dict({
        'model.test_cfg.mode': 'slide', 
        'model.test_cfg.crop_size': (512, 512), 
        'model.test_cfg.stride': (384, 384)
    })

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # slide window and crop image
    cnt = 0
    for data in tqdm.tqdm(data_loader, desc="Processing"):
        img = data['img'][0]
        img_meta = data['img_metas'][0].data[0]
        cnt += slide_crop(img, img_meta, cfg.model, save_dir)
    print(f'Preprocess finished, {cnt} binary files generated.')


if __name__ == '__main__':

    args = parse_args()
    preprocess(args.config, args.save_dir)
