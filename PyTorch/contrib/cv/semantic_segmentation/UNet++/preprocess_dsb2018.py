# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
import argparse
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data-science-bowl-2018',
                        help='the path of data-science-bowl-2018')
    parser.add_argument('--save_files', default='./inputs',
                        help='the path of data-science-bowl-2018')
    config = parser.parse_args()
    return config

def main(config):
    img_size = 96
    paths = glob(os.path.join(config.data_dir, 'stage1_train/*'))

    os.makedirs(config.save_files, exist_ok=True)
    images_path = os.path.join(config.save_files, './dsb2018_%d/images' % img_size)
    masks_path = os.path.join(config.save_files, './dsb2018_%d/masks/0' % img_size)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = cv2.imread(os.path.join(path, 'images',
                         os.path.basename(path) + '.png'))
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))
        cv2.imwrite(os.path.join(images_path, os.path.basename(path) + '.png'), img)
        cv2.imwrite(os.path.join(masks_path, os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    config = parse_args()
    main(config)
