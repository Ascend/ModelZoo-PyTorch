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
from collections import defaultdict
import csv
import os
import os.path as osp

from PIL import Image
import numpy as np
from tqdm import tqdm

from open_clip import get_model_config, get_tokenizer, image_transform


def data_preprcoess(model_name, data_dir, save_dir):
    model_cfg = get_model_config(model_name)
    image_size = model_cfg['vision_cfg']['image_size']
    image_preprocess = image_transform(
        (image_size, image_size),
        is_train=False,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    tokenizer = get_tokenizer(model_name)

    data_csv = osp.join(data_dir, 'flickr30k_test_karpathy.txt')
    image_save_dir = osp.join(save_dir, 'images')
    text_save_dir = osp.join(save_dir, 'texts')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(text_save_dir, exist_ok=True)

    text_counter = defaultdict(int)
    with open(data_csv, 'r') as f:
        csv_reader = csv.reader(f)
        for i, row in tqdm(enumerate(csv_reader)):
            if i == 0:
                continue   # skip table header
            image_name, text = [field.strip() for field in row]
            prefix = osp.splitext(image_name)[0]
            text_cnt = text_counter[image_name]

            # process image
            if text_cnt == 0:
                image_path = osp.join(data_dir, image_name)
                image_data = image_preprocess(Image.open(image_path)).unsqueeze(0)
                np.save(osp.join(image_save_dir, f'{prefix}.npy'), image_data)

            # process text
            text_data = tokenizer([text])
            np.save(osp.join(text_save_dir, f'{prefix}_{text_cnt}.npy'), text_data)
            text_counter[image_name] += 1


def main():
    parser = argparse.ArgumentParser(description='Data preprcoess.')
    parser.add_argument('--model-name', type=str,
                        default='ViT-B-32',
                        help='Specify the model name.')
    parser.add_argument('--data-dir', type=str,
                        default='./flickr30k_test1k',
                        help='Path to test dataset directory, which contains '
                             'test original images and text information file.')
    parser.add_argument('--save-dir', type=str,
                        default='./prep_data',
                        help='A directory to save preprocessed data.')
    args = parser.parse_args()

    data_preprcoess(args.model_name, args.data_dir, args.save_dir)


if __name__ == '__main__':
    main()
