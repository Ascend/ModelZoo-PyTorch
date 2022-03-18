# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import shutil

import cv2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

def main():

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file = open(val_ids_path)
    val_ids = file.read().split('\n')

    val_transform = Compose([
        transforms.Resize(96, 96),
        transforms.Normalize(),
    ])

    for img_id in val_ids:
        if len(img_id) == 0: continue
        img = cv2.imread(os.path.join(input_dir, img_id + '.png'))
        augmented = val_transform(image=img)
        img = augmented['image']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img.tofile(os.path.join(output_dir, img_id + ".bin"))

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    val_ids_path = sys.argv[3]
    main()
