# Copyright 2023 Huawei Technologies Co., Ltd
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
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from mmcv import Config

from mmocr.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(description='data preprocess.')
    parser.add_argument('--config', type=str, help='Test config file path.')
    parser.add_argument('--save-dir', type=str, 
                        help='a directory to save binary files.')
    args = parser.parse_args()
    preprocess(args.config, args.save_dir)


def create_mask(texts):
    num_text, num_char = texts.size()
    last_char_ids = (texts > 0).sum(-1) - 1
    valid_ids = torch.where(last_char_ids >= 0)[0]
    mask = torch.zeros((num_text, num_char, 256))
    mask[valid_ids, last_char_ids[valid_ids], :] = 1
    return mask


def preprocess(config_path, save_dir):

    save_dir = Path(save_dir)
    relations_dir = save_dir / 'relations'
    texts_dir = save_dir / 'texts'
    mask_dir = save_dir / 'mask'
    relations_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    num_data = len(dataset)

    for i in tqdm(range(num_data)):
        data = dataset[i]
        relations = data['relations'].data
        texts = data['texts'].data
        mask = create_mask(texts)
        
        img_name = data['img_metas'].data['ori_filename']
        npy_name = Path(img_name.replace('/', '-')).stem + '.npy'
        relations_path = relations_dir/npy_name
        texts_path = texts_dir/npy_name
        mask_path = mask_dir/npy_name

        np.save(relations_path, relations.numpy().astype(np.float32))
        np.save(texts_path, texts.numpy().astype(np.int32))
        np.save(mask_path, mask.numpy().astype(np.float32))


if __name__ == '__main__':
    main()
